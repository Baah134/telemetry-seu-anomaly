"""Summarize results/faults/*.json into tables and figures.

Per trial: counts are summed over the subset channels, then event F1, point F1 and false
alarms per 1k steps are computed (micro). Per condition: mean, std, 5th/95th percentiles
over the 100 trials and the fraction of CATASTROPHIC trials:
    event F1 < 0.5 x clean event F1   or   false alarms/1k > 3 x clean + 0.1
Writes results/faults_summary.json, results/tables/fault_*.{csv,md}, figures/fault_*.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.style import COL_W, DBL_W, save, setup
from src.detect.metrics import aggregate

F = Path("results/faults")
KIND_LABEL = {"int8": "INT8 (integer model)", "f32": "float32"}
KIND_COLOR = {"int8": "#1f4e9c", "f32": "#d9822b"}
PROT_LABEL = {"none": "no protection", "clamp": "activation range clamp", "sanity": "output sanity check",
              "tmr_L2": "TMR, output layer", "tmr_L0": "TMR, input layer", "scrub": "CRC32 scrub, K=100",
              "combined": "clamp + TMR(out) + scrub + sanity"}


def trial_metrics(recs):
    a = aggregate([r["counts"] for r in recs])
    return {"event_f1": a["event"]["f1"], "point_f1": a["point"]["f1"], "fa_1k": a["false_alarms_per_1k"],
            "nonfinite_any": float(any(r.get("nonfinite", 0) > 0 for r in recs))}


def stats(trials, clean):
    f1 = np.array([t["event_f1"] for t in trials])
    fa = np.array([t["fa_1k"] for t in trials])
    cata = (f1 < 0.5 * clean["event_f1"]) | (fa > 3 * clean["fa_1k"] + 0.1)
    return {"n_trials": len(trials), "event_f1_mean": float(f1.mean()), "event_f1_std": float(f1.std(ddof=1)),
            "event_f1_p5": float(np.percentile(f1, 5)), "event_f1_p95": float(np.percentile(f1, 95)),
            "point_f1_mean": float(np.mean([t["point_f1"] for t in trials])),
            "fa_1k_mean": float(fa.mean()), "catastrophic_frac": float(cata.mean()),
            "nonfinite_trial_frac": float(np.mean([t["nonfinite_any"] for t in trials]))}


def load(name):
    p = F / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def overheads(clean):
    """Per-model overhead of each protection (mean over the subset's models)."""
    wb = {k: float(np.mean(clean[f"{k}_weight_bytes"])) for k in ("int8", "f32")}
    # input layer size: (W + n_cmd) x 32 weights; output layer: 16 weights (+1 bias)
    in_dim = np.mean([88 if b < 4000 else 118 for b in clean["int8_weight_bytes"]])
    word = {"int8": 1, "f32": 4}
    out = {}
    for k in ("int8", "f32"):
        w = word[k]
        out[k] = {
            "none": (0, 0),
            "clamp": (2 * 3 * (1 if k == "int8" else 4), 49),
            "sanity": (8, 2),
            "tmr_L2": (2 * 16 * w, 16 * 3),
            "tmr_L0": (2 * in_dim * 32 * w, in_dim * 32 * 3),
            "scrub": (6 * 4, wb[k] / 100.0),                 # CRC words; bytes checked per inference at K=100
            "combined": (2 * 3 * (1 if k == "int8" else 4) + 8 + 2 * 16 * w + 24, 49 + 2 + 48 + wb[k] / 100.0),
        }
    return out, wb


def main():
    clean_rec = load("clean")
    clean = {k: {"event_f1": clean_rec[k]["event"]["f1"], "fa_1k": clean_rec[k]["false_alarms_per_1k"],
                 "point_f1": clean_rec[k]["point"]["f1"]} for k in ("int8", "f32")}
    out = {"subset": clean_rec["subset"], "clean": clean,
           "n_events_subset": clean_rec["int8"]["event"]["tp"] + clean_rec["int8"]["event"]["fn"]}
    plt = setup()
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    # ---------------- sweep
    S = load("sweep")
    if S:
        cond = {}
        for (kind, k, tr), res in zip(S["jobs"], S["results"]):
            cond.setdefault((kind, k), []).append(trial_metrics(res))
        rows = []
        out["sweep"] = {}
        for (kind, k), trials in sorted(cond.items()):
            st = stats(trials, clean[kind])
            out["sweep"][f"{kind}|{k}"] = st
            rows.append({"model": KIND_LABEL[kind], "flipped bits": k, "trials": st["n_trials"],
                         "event F1 mean ± std": f"{st['event_f1_mean']:.3f} ± {st['event_f1_std']:.3f}",
                         "event F1 5th-95th pct": f"{st['event_f1_p5']:.3f}-{st['event_f1_p95']:.3f}",
                         "false alarms / 1k": f"{st['fa_1k_mean']:.3f}",
                         "catastrophic trials": f"{st['catastrophic_frac'] * 100:.0f}%",
                         "trials with NaN/inf forecasts": f"{st['nonfinite_trial_frac'] * 100:.0f}%"})
        df = pd.DataFrame(rows)
        df.to_csv("results/tables/fault_sweep.csv", index=False)
        Path("results/tables/fault_sweep.md").write_text(df.to_markdown(index=False) + (
            f"\n\nClean event F1: INT8 {clean['int8']['event_f1']:.3f}, float32 {clean['f32']['event_f1']:.3f} "
            f"(subset of {len(out['subset'])} channels, {out['n_events_subset']} labeled events). Flips persist for "
            "the whole test stream. Catastrophic = event F1 < 50% of clean or false alarms/1k > 3x clean + 0.1.\n"),
            encoding="utf-8")
        print(df.to_markdown(index=False))
        Dm = load("damage")
        dmg = {}
        if Dm:
            for (kind, k, tr), res in zip(Dm["jobs"], Dm["results"]):
                r = np.array([x["test_nonanom_mse_ratio"] for x in res], float)
                r = np.where(np.isfinite(r), r, np.inf)
                dmg.setdefault((kind, k), []).extend(r.tolist())
            out["damage"] = {}
            drows = []
            for (kind, k), r in sorted(dmg.items()):
                r = np.array(r)
                d = {"models": int(r.size), "frac_mse_gt_2x": float(np.mean(r > 2)),
                     "frac_mse_gt_10x": float(np.mean(r > 10)), "median_ratio": float(np.median(r)),
                     "event_f1_mean": out["sweep"][f"{kind}|{k}"]["event_f1_mean"]}
                out["damage"][f"{kind}|{k}"] = d
                drows.append({"model": KIND_LABEL[kind], "flipped bits": k, "models (100 trials x 10 ch.)": d["models"],
                              "forecast MSE > 2x clean": f"{d['frac_mse_gt_2x'] * 100:.0f}%",
                              "forecast MSE > 10x clean": f"{d['frac_mse_gt_10x'] * 100:.0f}%",
                              "event F1 (same flips)": f"{d['event_f1_mean']:.3f}"})
            df = pd.DataFrame(drows)
            df.to_csv("results/tables/fault_damage.csv", index=False)
            Path("results/tables/fault_damage.md").write_text(df.to_markdown(index=False) + (
                "\n\nForecast MSE on test points outside labeled anomalies, relative to the clean model, for the "
                "same flips as the sweep. Shows whether detection F1 tracks model health.\n"), encoding="utf-8")
            print(df.to_markdown(index=False))
        fig, axs = plt.subplots(1, 3, figsize=(DBL_W, 2.3))
        for kind in ("int8", "f32"):
            ks = sorted({k for kk, k in cond if kk == kind})
            mu = [out["sweep"][f"{kind}|{k}"]["event_f1_mean"] for k in ks]
            lo = [out["sweep"][f"{kind}|{k}"]["event_f1_p5"] for k in ks]
            hi = [out["sweep"][f"{kind}|{k}"]["event_f1_p95"] for k in ks]
            axs[0].plot(ks, mu, "o-", color=KIND_COLOR[kind], label=KIND_LABEL[kind], ms=3)
            axs[0].fill_between(ks, lo, hi, color=KIND_COLOR[kind], alpha=0.15, lw=0)
            axs[0].axhline(clean[kind]["event_f1"], color=KIND_COLOR[kind], ls=":", lw=0.8)
            axs[1].plot(ks, [out["sweep"][f"{kind}|{k}"]["nonfinite_trial_frac"] * 100 for k in ks], "o-",
                        color=KIND_COLOR[kind], ms=3)
            if dmg:
                axs[2].plot(ks, [out["damage"][f"{kind}|{k}"]["frac_mse_gt_2x"] * 100 for k in ks], "o-",
                            color=KIND_COLOR[kind], ms=3)
        for ax in axs:
            ax.set_xscale("log")
            ax.set_xlabel("bit flips in weight memory")
        axs[0].set_ylabel("event F1 (mean, 5-95% band)")
        axs[1].set_ylabel("trials with NaN/inf forecasts (%)")
        axs[2].set_ylabel("models with forecast MSE > 2x (%)")
        axs[0].legend(fontsize=6)
        fig.tight_layout()
        save(fig, "figures/fault_sweep")

    # ---------------- bit position / layer
    B = load("bitpos")
    if B:
        agg = {}
        for (kind, tr, l, b), res in zip(B["jobs"], B["results"]):
            for r in res:
                agg.setdefault((kind, l, b), []).append(r)
        out["bitpos"] = {}
        fig, axs = plt.subplots(2, 1, figsize=(DBL_W, 3.4))
        fig.subplots_adjust(hspace=0.75)
        for ax, (kind, nb) in zip(axs, (("int8", 8), ("f32", 32))):
            M = np.zeros((3, nb))
            for l in range(3):
                for b in range(nb):
                    rs = agg[(kind, l, b)]
                    M[l, b] = np.mean([r["val_dev_max"] > 0.1 for r in rs]) * 100
                    out["bitpos"][f"{kind}|L{l}|b{b}"] = {
                        "p_forecast_change_gt_0.1": M[l, b] / 100,
                        "median_dev": float(np.median([r["val_dev"] for r in rs])),
                        "nonfinite_frac": float(np.mean([r["val_nonfinite_frac"] > 0 for r in rs]))}
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=100)
            ax.set_yticks(range(3), ["L0 (input)", "L1", "L2 (output)"])
            ax.set_xticks(range(nb), [str(b) for b in range(nb)], fontsize=5 if nb > 8 else 7)
            ax.set_title(f"{KIND_LABEL[kind]}: % of single flips that change a validation forecast by > 0.1 sd "
                         f"(bit {nb - 1} = sign)")
            fig.colorbar(im, ax=ax, pad=0.01)
        axs[1].set_xlabel("bit position in the stored weight word (0 = LSB)")
        save(fig, "figures/fault_bitpos")

    # ---------------- transient
    T = load("transient")
    if T:
        cond = {}
        for (kind, p, tr), res in zip(T["jobs"], T["results"]):
            cond.setdefault((kind, p), []).append(trial_metrics(res))
        out["transient"] = {f"{k}|{p}": stats(v, clean[k]) for (k, p), v in sorted(cond.items())}
        rows = [{"model": KIND_LABEL[k], "upset prob. per inference": p, "event F1 mean ± std":
                 f"{out['transient'][f'{k}|{p}']['event_f1_mean']:.3f} ± {out['transient'][f'{k}|{p}']['event_f1_std']:.3f}",
                 "false alarms / 1k": f"{out['transient'][f'{k}|{p}']['fa_1k_mean']:.3f}",
                 "catastrophic trials": f"{out['transient'][f'{k}|{p}']['catastrophic_frac'] * 100:.0f}%",
                 "trials with NaN/inf": f"{out['transient'][f'{k}|{p}']['nonfinite_trial_frac'] * 100:.0f}%"}
                for (k, p) in sorted(cond)]
        df = pd.DataFrame(rows)
        df.to_csv("results/tables/fault_transient.csv", index=False)
        Path("results/tables/fault_transient.md").write_text(df.to_markdown(index=False) + (
            "\n\nOne random bit flip in a random int32 accumulator (INT8) or float32 pre-activation (float) "
            "with the given probability per inference.\n"), encoding="utf-8")
        print(df.to_markdown(index=False))

    # ---------------- protections
    P = load("protect")
    if P:
        ov, wb = overheads(clean_rec)
        cond = {}
        for (kind, k, tr), res in zip(P["jobs"], P["results"]):
            for prot, recs in res.items():
                cond.setdefault((kind, k, prot), []).append(trial_metrics(recs))
        out["protect"] = {f"{a}|{b}|{c}": stats(v, clean[a]) for (a, b, c), v in cond.items()}
        rows = []
        for kind in ("int8", "f32"):
            for prot in PROT_LABEL:
                row = {"model": KIND_LABEL[kind], "protection": PROT_LABEL[prot],
                       "extra memory [bytes/model]": f"{ov[kind][prot][0]:.0f}",
                       "extra work / inference (approx. ops or bytes read)": f"{ov[kind][prot][1]:.0f}"}
                for k in sorted({b for a, b, c in cond if a == kind}):
                    st = out["protect"].get(f"{kind}|{k}|{prot}")
                    if st:
                        row[f"event F1, k={k}"] = f"{st['event_f1_mean']:.3f} ± {st['event_f1_std']:.3f}"
                        row[f"catastrophic, k={k}"] = f"{st['catastrophic_frac'] * 100:.0f}%"
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv("results/tables/fault_protect.csv", index=False)
        Path("results/tables/fault_protect.md").write_text(df.to_markdown(index=False) + (
            f"\n\nk flips at a uniform random time t0 in the test stream; persist afterwards unless scrubbed. "
            f"Mean weight memory per model: INT8 {wb['int8']:.0f} B, float32 {wb['f32']:.0f} B. Overheads are "
            "analytic estimates per model (see docs/EXPLAINER.md section 6). Clean event F1: INT8 "
            f"{clean['int8']['event_f1']:.3f}, float32 {clean['f32']['event_f1']:.3f}.\n"), encoding="utf-8")
        print(df.to_markdown(index=False))
        fig, axs = plt.subplots(1, 2, figsize=(DBL_W, 2.6), sharey=True)
        kmax = max(b for a, b, c in cond)
        for ax, kind in zip(axs, ("int8", "f32")):
            for prot in PROT_LABEL:
                st = out["protect"].get(f"{kind}|{kmax}|{prot}")
                if not st:
                    continue
                x = max(ov[kind][prot][0], 1)
                ax.scatter(x, st["event_f1_mean"], s=14, zorder=3)
                ax.annotate(PROT_LABEL[prot], (x, st["event_f1_mean"]), fontsize=5, xytext=(3, 2),
                            textcoords="offset points")
            ax.axhline(clean[kind]["event_f1"], color="k", ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("extra memory per model (bytes, log; 'none' plotted at 1)")
            ax.set_title(f"{KIND_LABEL[kind]}, k = {kmax} flips")
        axs[0].set_ylabel("event F1 under faults (dotted: clean)")
        save(fig, "figures/fault_protect")
        # scrub interval
        ks = sorted(int(c.split("@")[1]) for a, b, c in cond if c.startswith("scrub@"))
        if ks:
            sub = {K: cond[("int8", 10, f"scrub@{K}")] for K in ks}
            corr = {}
            for (kind, k, tr), res in zip(P["jobs"], P["results"]):
                if kind == "int8" and k == 10:
                    for K in ks:
                        corr.setdefault(K, []).extend(r["corrupt_inferences"] for r in res[f"scrub@{K}"])
            fig, ax = plt.subplots(figsize=(COL_W, 2.2))
            ax.plot(ks, [np.mean([t["event_f1"] for t in sub[K]]) for K in ks], "o-", ms=3, label="event F1")
            ax.set_xscale("log")
            ax.set_xlabel("checksum interval K (inferences)")
            ax.set_ylabel("event F1 (INT8, k = 10)")
            ax2 = ax.twinx()
            ax2.plot(ks, [np.mean(corr[K]) for K in ks], "s--", ms=3, color="tab:red", label="corrupted inferences")
            ax2.set_yscale("log")
            ax2.set_ylabel("mean corrupted inferences\nbefore repair", color="tab:red")
            ax.set_title(f"CRC cost: {wb['int8']:.0f}/K bytes read per inference")
            save(fig, "figures/fault_scrub_interval")
            out["scrub_interval"] = {str(K): {"event_f1_mean": float(np.mean([t["event_f1"] for t in sub[K]])),
                                              "mean_corrupt_inferences": float(np.mean(corr[K])),
                                              "bytes_checked_per_inference": wb["int8"] / K} for K in ks}
    Path("results/faults_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
