"""Simulated SEU fault-injection study on the tiny forecaster (INT8 integer model and its
float32 counterpart). Simulation only.

Channel subset: FAULT_N_PER_SC channels per spacecraft drawn at random (seed 0) from the
channels with at least one labeled anomaly; fixed before any fault results are seen.
Model: training seed 0 of each channel. Threshold: the clean model's validation-fitted
percentile threshold (stored in protected memory, not subject to faults).

Experiments (each condition: N_TRIALS Monte Carlo trials; one trial = an independent
fault draw in EVERY subset channel's model, metrics aggregated over the subset):
  sweep      k weight-memory bit flips (k in K_LIST), faults present for the whole test
             stream (worst case), int8 vs float32, no protection
  bitpos     1 flip restricted to one (layer, bit position) of the weight matrices;
             label-free sensitivity = median relative change of validation forecasts
             (used to pick the TMR layer), plus test event-F1
  transient  per-inference probability p of one bit flip in a random int32 accumulator
             (int8) / float32 pre-activation (float), p in P_LIST
  protect    k in K_PROTECT flips injected at a uniform random time t0 in the test stream
             (same t0 draw for every protection); protections: none, clamp, scrub@K,
             tmr(layer), sanity, combined
Writes results/faults/<experiment>.json.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path("results/faults")
FAULT_N_PER_SC = 5
N_TRIALS = 100
K_LIST = [1, 2, 5, 10, 20, 50, 100]
P_LIST = [1e-4, 1e-3, 1e-2, 1e-1]
K_PROTECT = [1, 10, 100]
SCRUB_K = [1, 10, 100, 1000]
CATA_F1_FRAC = 0.5      # catastrophic: event F1 < 50% of clean ...
CATA_FA_MULT = 3.0      # ... or false-alarm events per 1k > 3x clean (+0.1)


def pick_subset(seed=0):
    import pandas as pd
    from src.data.loader import load_labels
    df = load_labels()
    df = df[df.anomaly_sequences.apply(len) > 0]
    rng = np.random.default_rng(seed)
    out = []
    for sc in ("SMAP", "MSL"):
        ids = sorted(df[df.spacecraft == sc].chan_id)
        out += sorted(rng.choice(ids, size=min(FAULT_N_PER_SC, len(ids)), replace=False).tolist())
    return out


class Ctx:
    """Everything one channel needs for fault trials (built once per worker)."""

    def __init__(self, chan_id):
        import torch
        from src.data.loader import W, load_channel, load_labels, windows
        from src.detect.threshold import percentile_threshold, smoothed_error
        from src.models.nets import TinyMLP
        from src.quant.int8 import activation_ranges, float_layers, quantize
        row = next(r for r in load_labels().to_dict("records") if r["chan_id"] == chan_id)
        ch = load_channel(row)
        n_fit = len(ch.train) - int(round(0.2 * len(ch.train)))
        Xall, yall = windows(ch.train, ch)
        tgt = np.arange(W, len(ch.train))
        self.Xf, self.Xv, self.yv = Xall[tgt < n_fit], Xall[tgt >= n_fit], yall[tgt >= n_fit]
        self.yf = yall[tgt < n_fit]
        self.Xt, self.yt = windows(ch.test, ch)
        self.n, self.W, self.labels = len(ch.test), W, ch.label_seqs
        ck = torch.load(Path("results/models") / f"{chan_id}_s0.pt")
        m = TinyMLP(ck["in_dim"])
        m.load_state_dict(ck["tiny"])
        self.layers = float_layers(m)
        self.qm = quantize(self.layers, self.Xf)
        r = activation_ranges(self.layers, self.Xf)
        self.fclamp = [(0.0, r[1]), (0.0, r[2]), (-r[3], r[3])]
        # int8 clamp: profiled max code per layer from calibration data
        self.iclamp = self._int_ranges()
        self.lo, self.hi = float(self.yf.min()), float(self.yf.max())
        self.smoothed_error = smoothed_error
        self.eps = {}
        self.clean = {}
        for kind in ("int8", "f32"):
            pv = self.forward(kind, self.Xv, self.weights(kind))
            self.eps[kind] = percentile_threshold(smoothed_error(self.yv, pv))
            self.clean[kind] = self.forward(kind, self.Xt, self.weights(kind))

    def _int_ranges(self):
        from src.quant.int8 import QMAX, requant
        a = self.qm.quantize_input(self.Xf).astype(np.int64)
        out = []
        L = len(self.qm.W)
        for l in range(L):
            acc = (a.astype(np.float64) @ self.qm.W[l].astype(np.float64).T).astype(np.int64) + self.qm.b[l]
            y = np.clip(requant(acc, self.qm.M0[l], self.qm.sh[l]), 0 if l < L - 1 else -QMAX, QMAX)
            out.append((int(y.min()), int(y.max())))
            a = y
        return out

    def weights(self, kind):
        """Stored words: [W1, b1, W2, b2, W3, b3]."""
        if kind == "int8":
            return [x for l in range(3) for x in (self.qm.W[l], self.qm.b[l])]
        return [x for W, b in self.layers for x in (W, b)]

    def forward(self, kind, X, ws, clamp=False, hook=None):
        if kind == "int8":
            q = self.qm.copy()
            q.W, q.b = [ws[0], ws[2], ws[4]], [ws[1], ws[3], ws[5]]
            return q.predict(X, act_hook=hook, act_clamp=self.iclamp if clamp else None)
        from src.quant.int8 import float_forward
        layers = [(ws[0], ws[1]), (ws[2], ws[3]), (ws[4], ws[5])]
        if hook is None:
            with np.errstate(all="ignore"):
                return float_forward(layers, X, self.fclamp if clamp else None).astype(np.float64)
        a = X.astype(np.float32)
        with np.errstate(all="ignore"):
            for l, (Wl, bl) in enumerate(layers):
                a = hook(l, a @ Wl.T + bl)
                if l < 2:
                    a = np.maximum(a, 0)
                if clamp:
                    a = np.clip(a, *self.fclamp[l])
        return a[:, 0].astype(np.float64)

    def counts(self, kind, pred):
        from src.detect.metrics import all_counts
        s = np.zeros(self.n)
        with np.errstate(all="ignore"):
            s[self.W:] = self.smoothed_error(self.yt, pred)
        alarm = np.zeros(self.n, bool)
        alarm[self.W:] = s[self.W:] > self.eps[kind]      # NaN scores never alarm (as on an MCU)
        return all_counts(alarm, self.labels), int(np.sum(~np.isfinite(pred)))


_CTX = {}


def ctx(c):
    if c not in _CTX:
        _CTX[c] = Ctx(c)
    return _CTX[c]


# ---------------------------------------------------------------- trial functions
def trial_weights(args):
    """Persistent weight flips present for the whole stream (sweep / bitpos)."""
    from src.faults.inject import flip_bits
    chans, kind, k, trial, layer, bit = args
    import warnings
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng([trial, k, 0 if kind == "int8" else 1, 0 if layer is None else layer + 1,
                                 0 if bit is None else bit + 1])
    out = []
    for c in chans:
        cx = ctx(c)
        ws = cx.weights(kind)
        if layer is None:
            new, _ = flip_bits(ws, k, rng)
        else:
            sub, _ = flip_bits([ws[2 * layer]], k, rng, bit_positions=[bit])
            new = list(ws)
            new[2 * layer] = sub[0]
        pt = cx.forward(kind, cx.Xt, new)
        cnt, nonfinite = cx.counts(kind, pt)
        rec = {"chan": c, "counts": cnt, "nonfinite": nonfinite}
        if layer is not None:      # label-free sensitivity on validation windows
            pv0 = cx.forward(kind, cx.Xv, ws)
            pv = cx.forward(kind, cx.Xv, new)
            with np.errstate(all="ignore"):
                d = np.abs(pv - pv0)
            rec["val_dev"] = float(np.median(np.where(np.isfinite(d), d, 1e6)))
            rec["val_dev_max"] = float(np.max(np.where(np.isfinite(d), d, 1e6)))
        out.append(rec)
    return out


def trial_transient(args):
    from src.faults.inject import flip_int32_bit
    chans, kind, p, trial = args
    rng = np.random.default_rng([trial, int(round(-np.log10(p) * 10)), 7 if kind == "int8" else 8])
    out = []
    for c in chans:
        cx = ctx(c)
        N = len(cx.Xt)
        hit = rng.random(N) < p
        rows = np.nonzero(hit)[0]
        lay = rng.integers(0, 3, rows.size)
        sizes = [w.shape[0] for w in cx.qm.W]
        neu = np.array([rng.integers(0, sizes[l]) for l in lay], dtype=np.int64)
        bits = rng.integers(0, 32, rows.size)

        def hook(l, acc):
            sel = lay == l
            if not sel.any():
                return acc
            acc = acc.copy()
            r, n_, b = rows[sel], neu[sel], bits[sel]
            if kind == "int8":
                for ri, ni, bi in zip(r, n_, b):
                    acc[ri, ni] = flip_int32_bit(np.array([acc[ri, ni]]), int(bi))[0]
            else:
                v = acc[r, n_].astype(np.float32).copy()
                u = v.view(np.uint32)
                u ^= (np.uint32(1) << b.astype(np.uint32))
                acc[r, n_] = u.view(np.float32)
            return acc

        pt = cx.forward(kind, cx.Xt, cx.weights(kind), hook=hook)
        cnt, nonfinite = cx.counts(kind, pt)
        out.append({"chan": c, "counts": cnt, "nonfinite": nonfinite, "n_upsets": int(rows.size)})
    return out


PROTECTIONS = ["none", "clamp", "sanity", "tmr", "scrub", "combined"]


def trial_protect(args):
    """k flips at random time t0; each protection evaluated on the SAME fault draw."""
    from src.faults.inject import flip_bits
    from src.faults.protect import sanity_filter, tmr_vote
    chans, kind, k, trial, tmr_layer, scrub_k = args
    rng = np.random.default_rng([trial, k, 11 if kind == "int8" else 12])
    res = {p: [] for p in PROTECTIONS + [f"scrub@{K}" for K in SCRUB_K]}
    for c in chans:
        cx = ctx(c)
        ws = cx.weights(kind)
        N = len(cx.Xt)
        t0 = int(rng.integers(0, N))
        # memory with TMR on one layer's weight matrix: 3 copies, flips uniform over all bits
        tw = 2 * tmr_layer
        mem = list(ws) + [ws[tw].copy(), ws[tw].copy()]
        new, _ = flip_bits(mem, k, rng)
        plain = new[:6]
        # without TMR the extra copies do not exist, so the unprotected memory gets its own
        # k-flip draw (same k, same t0); TMR trials are therefore not bit-for-bit paired.
        plain_noTMR, _ = flip_bits(ws, k, rng)
        voted = list(plain)
        voted[tw] = tmr_vote(new[tw], new[6], new[7])
        clean = cx.clean[kind]
        last = cx.Xt[:, cx.W - 1].astype(np.float64)

        def stream(corrupt_pred, until=None):
            p = clean.copy()
            end = N if until is None else min(N, until)
            p[t0:end] = corrupt_pred[t0:end]
            return p

        f_none = cx.forward(kind, cx.Xt, plain_noTMR)
        f_clamp = cx.forward(kind, cx.Xt, plain_noTMR, clamp=True)
        f_tmr = cx.forward(kind, cx.Xt, voted)
        f_comb = cx.forward(kind, cx.Xt, voted, clamp=True)
        preds = {
            "none": stream(f_none),
            "clamp": stream(f_clamp),
            "sanity": sanity_filter(stream(f_none), last, cx.lo, cx.hi)[0],
            "tmr": stream(f_tmr),
            # scrub: detected at the next multiple of K after t0, then golden weights reloaded
            "scrub": stream(f_none, until=(t0 // scrub_k + 1) * scrub_k),
            "combined": sanity_filter(stream(f_comb, until=(t0 // scrub_k + 1) * scrub_k), last, cx.lo, cx.hi)[0],
        }
        for K in SCRUB_K:
            preds[f"scrub@{K}"] = stream(f_none, until=(t0 // K + 1) * K)
        for name, p in preds.items():
            cnt, nonfinite = cx.counts(kind, p)
            res[name].append({"chan": c, "counts": cnt, "nonfinite": nonfinite,
                              "corrupt_inferences": int(min(N, (t0 // scrub_k + 1) * scrub_k) - t0)
                              if "scrub" in name or name == "combined" else int(N - t0)})
    return res


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings
    warnings.filterwarnings("ignore")


def run(ex, fn, jobs):
    return list(ex.map(fn, jobs, chunksize=max(1, len(jobs) // 200)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--only", nargs="*", default=["clean", "sweep", "bitpos", "transient", "protect"])
    ap.add_argument("--tmr-layer", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    chans = pick_subset()
    T = a.trials
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        if "clean" in a.only:
            from src.detect.metrics import aggregate
            rec = {"subset": chans}
            for kind in ("int8", "f32"):
                cnts = [ctx(c).counts(kind, ctx(c).clean[kind])[0] for c in chans]
                rec[kind] = aggregate(cnts)
                rec[f"{kind}_weight_bytes"] = [int(sum(w.nbytes for w in ctx(c).weights(kind))) for c in chans]
            (OUT / "clean.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print("clean", {k: rec[k]["event"]["f1"] for k in ("int8", "f32")}, flush=True)
        if "sweep" in a.only:
            t = time.time()
            jobs = [(chans, kind, k, tr, None, None) for kind in ("int8", "f32") for k in K_LIST for tr in range(T)]
            res = run(ex, trial_weights, jobs)
            (OUT / "sweep.json").write_text(json.dumps(
                {"subset": chans, "jobs": [j[1:4] for j in jobs], "results": res}), encoding="utf-8")
            print(f"sweep {time.time() - t:.0f}s", flush=True)
        if "bitpos" in a.only:
            t = time.time()
            jobs = [(chans, kind, 1, tr, l, b) for kind, nb in (("int8", 8), ("f32", 32))
                    for l in range(3) for b in range(nb) for tr in range(T)]
            res = run(ex, trial_weights, jobs)
            (OUT / "bitpos.json").write_text(json.dumps(
                {"subset": chans, "jobs": [(j[1], j[3], j[4], j[5]) for j in jobs], "results": res}),
                encoding="utf-8")
            print(f"bitpos {time.time() - t:.0f}s", flush=True)
        if "transient" in a.only:
            t = time.time()
            jobs = [(chans, kind, p, tr) for kind in ("int8", "f32") for p in P_LIST for tr in range(T)]
            res = run(ex, trial_transient, jobs)
            (OUT / "transient.json").write_text(json.dumps(
                {"subset": chans, "jobs": [j[1:] for j in jobs], "results": res}), encoding="utf-8")
            print(f"transient {time.time() - t:.0f}s", flush=True)
        if "protect" in a.only:
            if a.tmr_layer is None:
                raise SystemExit("--tmr-layer required (pick from bitpos sensitivity, label-free)")
            t = time.time()
            jobs = [(chans, kind, k, tr, a.tmr_layer, 100) for kind in ("int8", "f32") for k in K_PROTECT
                    for tr in range(T)]
            res = run(ex, trial_protect, jobs)
            (OUT / "protect.json").write_text(json.dumps(
                {"subset": chans, "tmr_layer": a.tmr_layer, "scrub_k_main": 100, "scrub_k": SCRUB_K,
                 "jobs": [j[1:4] for j in jobs], "results": res}), encoding="utf-8")
            print(f"protect {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
