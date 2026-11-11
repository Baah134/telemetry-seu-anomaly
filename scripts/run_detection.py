"""Train and evaluate every detector on every channel.

Usage: python -m scripts.run_detection [--phase fast|lstm] [--seeds 0 1 2] [--workers 11]
Writes results/detection/<chan>.json (counts per method/seed/threshold) and
results/models/<chan>_s<seed>.pt (tiny + LSTM weights; gitignored, regenerated).

Methods (all share windows, normalization, scoring and thresholds):
  tiny_f32   TinyMLP forecaster, float32
  tiny_int8  same weights after INT8 PTQ, run by the integer-only NumPy reference
  lstm       2x80 LSTM forecaster, float32 (reference)
  persist    persistence forecast y_hat[t] = y[t-1]
  zscore     rolling z-score of the raw value over the previous W points (no forecast)
  random     i.i.d. uniform scores (seeded), for the point-adjust inflation check
Scores: EWMA (alpha 0.05) of |error| (forecasters) or of the z-score; random is raw.
Thresholds fitted on the validation split (end of train) only: percentile (99.5) and ndt.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

OUT = Path("results/detection")
MODELS = Path("results/models")


def rolling_z(v, w):
    """|v[t] - mean(v[t-w:t])| / std(v[t-w:t]) for t >= w."""
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(v, w)[:-1]
    mu, sd = win.mean(1), win.std(1)
    return np.abs(v[w:] - mu) / np.maximum(sd, 1e-3)


def _job(chan_id, seeds, epochs_lstm, phase):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    from src.data.loader import W, load_labels, load_channel, windows
    from src.detect.metrics import all_counts
    from src.detect.telemanom import detect as tm_detect
    from src.detect.threshold import THRESHOLDS, ewma, smoothed_error
    from src.models.nets import LSTMRef, TinyMLP, n_params, predict, train_forecaster
    from src.quant.int8 import float_layers, quantize

    t0 = time.perf_counter()
    row = next(r for r in load_labels().to_dict("records") if r["chan_id"] == chan_id)
    ch = load_channel(row)
    n_fit = len(ch.train) - int(round(0.2 * len(ch.train)))
    Xall, yall = windows(ch.train, ch)
    tgt_idx = np.arange(W, len(ch.train))
    fit, val = tgt_idx < n_fit, tgt_idx >= n_fit
    Xf, yf, Xv, yv = Xall[fit], yall[fit], Xall[val], yall[val]
    Xt, yt = windows(ch.test, ch)
    n_test = len(ch.test)

    jf = OUT / f"{chan_id}.json"
    if phase == "lstm":
        rec = json.loads(jf.read_text(encoding="utf-8"))
    else:
        rec = {"chan_id": chan_id, "spacecraft": ch.spacecraft, "n_test": n_test, "n_train": len(ch.train),
               "n_cmd": ch.n_cmd, "label_seqs": ch.label_seqs, "classes": ch.classes, "methods": {}}

    def evaluate(name, seed, s_val, s_test, extra=None, test_pred=None):
        out = {}
        for tn, tf in THRESHOLDS.items():
            eps = tf(s_val)
            pred = np.zeros(n_test, bool)
            pred[W:] = s_test > eps
            out[tn] = {"eps": eps, "counts": all_counts(pred, ch.label_seqs), "alarm_frac": float(pred.mean())}
        if test_pred is not None:
            # telemanom's own pipeline (label-free, windowed over test errors), original units
            e_raw = np.abs(yt - test_pred) * ch.sd
            seqs = tm_detect(e_raw, yt * ch.sd + ch.mu, l_s=W, chan_id=chan_id)
            pred = np.zeros(n_test, bool)
            for s, e in seqs:
                pred[s + W:e + W + 1] = True
            out["telemanom"] = {"counts": all_counts(pred, ch.label_seqs), "alarm_frac": float(pred.mean()),
                                "n_seqs": len(seqs)}
        rec["methods"].setdefault(name, {})[str(seed)] = {"thr": out, **(extra or {})}

    def fc_scores(pv, pt):
        return smoothed_error(yv, pv), smoothed_error(yt, pt)

    MODELS.mkdir(parents=True, exist_ok=True)
    if phase == "lstm":
        for seed in seeds:
            lstm, vl2 = train_forecaster(LSTMRef(W, ch.n_cmd), Xf, yf, Xv, yv, seed, epochs=epochs_lstm,
                                         batch=128, lr=2e-3, patience=4)
            lv, lt = predict(lstm, Xv), predict(lstm, Xt)
            evaluate("lstm", seed, *fc_scores(lv, lt), {"val_mse": vl2, "params": n_params(lstm),
                                                        "test_mse": float(np.mean((lt - yt) ** 2))}, test_pred=lt)
            torch.save(lstm.state_dict(), MODELS / f"{chan_id}_s{seed}_lstm.pt")
        rec["wall_s_lstm"] = time.perf_counter() - t0
        jf.write_text(json.dumps(rec), encoding="utf-8")
        return chan_id, rec["wall_s_lstm"]

    # deterministic baselines (seed-independent)
    evaluate("persist", 0, *fc_scores(Xv[:, W - 1], Xt[:, W - 1]), test_pred=Xt[:, W - 1])
    vtr = (ch.train[:, 0] - ch.mu) / ch.sd
    vte = (ch.test[:, 0] - ch.mu) / ch.sd
    z_tr = rolling_z(vtr, W)
    evaluate("zscore", 0, ewma(z_tr[val]), ewma(rolling_z(vte, W)))
    for seed in seeds:
        rng = np.random.default_rng(10_000 + seed)
        evaluate("random", seed, rng.random(val.sum()), rng.random(len(yt)))

    for seed in seeds:
        tiny, vl = train_forecaster(TinyMLP(Xf.shape[1]), Xf, yf, Xv, yv, seed)
        pv, pt = predict(tiny, Xv), predict(tiny, Xt)
        evaluate("tiny_f32", seed, *fc_scores(pv, pt), {"val_mse": vl, "params": n_params(tiny),
                                                        "test_mse": float(np.mean((pt - yt) ** 2))}, test_pred=pt)
        qm = quantize(float_layers(tiny), Xf)
        qv, qt = qm.predict(Xv), qm.predict(Xt)
        evaluate("tiny_int8", seed, *fc_scores(qv, qt), {"weight_bytes": qm.n_weight_bytes,
                                                         "test_mse": float(np.mean((qt - yt) ** 2)),
                                                         "max_abs_vs_f32": float(np.max(np.abs(qt - pt)))}, test_pred=qt)
        torch.save({"tiny": tiny.state_dict(), "in_dim": Xf.shape[1]}, MODELS / f"{chan_id}_s{seed}.pt")
    rec["wall_s"] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    jf.write_text(json.dumps(rec), encoding="utf-8")
    return chan_id, rec["wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--channels", nargs="*")
    ap.add_argument("--epochs-lstm", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--phase", choices=["fast", "lstm"], default="fast",
                    help="fast = baselines + tiny models; lstm = add the LSTM reference to existing results")
    a = ap.parse_args()
    from src.data.loader import load_labels
    chans = a.channels or list(load_labels().chan_id)
    def done(c):
        f = OUT / f"{c}.json"
        if not f.exists():
            return False
        return a.phase == "fast" or "lstm" in json.loads(f.read_text(encoding="utf-8"))["methods"]
    todo = [c for c in chans if a.force or not done(c)]
    if a.phase == "lstm":
        todo = [c for c in todo if (OUT / f"{c}.json").exists()]
    print(f"{len(todo)} channels, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_job, c, a.seeds, a.epochs_lstm, a.phase) for c in todo]
        for f in as_completed(futs):
            c, w = f.result()
            print(f"done {c} in {w:.0f}s", flush=True)


if __name__ == "__main__":
    main()
