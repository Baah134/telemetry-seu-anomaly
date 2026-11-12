"""Aggregate results/detection/*.json into the detection tables and figures.

Per method x threshold x spacecraft (SMAP, MSL, ALL): micro-aggregated event-wise,
point-wise and point-adjusted P/R/F1 and false-alarm events per 1k timesteps, computed
per training seed and reported as mean +- std over seeds (deterministic methods: 1 seed).
Writes results/detection_summary.json, results/tables/detection_<thr>.{csv,md},
results/tables/benchmark_sanity.{csv,md}, figures/detection_f1, figures/pa_inflation.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.style import COL_W, COLORS, DBL_W, LABELS, METHODS, save, setup
from src.detect.metrics import aggregate

THRS = ["telemanom", "percentile", "ndt"]
GROUPS = ["SMAP", "MSL", "ALL"]


def load():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path("results/detection").glob("*.json"))]


def summarize(recs):
    out = {}
    for m in METHODS:
        for t in THRS:
            for g in GROUPS:
                rs = [r for r in recs if g == "ALL" or r["spacecraft"] == g]
                seeds = sorted({s for r in rs for s in r["methods"].get(m, {})})
                per_seed = []
                for s in seeds:
                    cnts = [r["methods"][m][s]["thr"][t]["counts"] for r in rs
                            if s in r["methods"].get(m, {}) and t in r["methods"][m][s]["thr"]]
                    if len(cnts) < len(rs):
                        continue
                    per_seed.append(aggregate(cnts))
                if not per_seed:
                    continue
                vals = defaultdict(list)
                for a in per_seed:
                    for k in ("event", "point", "pa"):
                        for q in ("precision", "recall", "f1"):
                            vals[f"{k}_{q}"].append(a[k][q])
                    vals["fa_per_1k"].append(a["false_alarms_per_1k"])
                    vals["events_detected"].append(a["event"]["tp"])
                    vals["events_total"].append(a["event"]["tp"] + a["event"]["fn"])
                out[f"{m}|{t}|{g}"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                           "per_seed": v} for k, v in vals.items()}
                out[f"{m}|{t}|{g}"]["n_seeds"] = len(per_seed)
                out[f"{m}|{t}|{g}"]["n_channels"] = len(rs)
    return out


def fmt(d, k, nd=3):
    x = d[k]
    return f"{x['mean']:.{nd}f}" if d["n_seeds"] == 1 else f"{x['mean']:.{nd}f} ± {x['std']:.{nd}f}"


def tables(S):
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    for t in THRS:
        rows = []
        for g in GROUPS:
            for m in METHODS:
                d = S.get(f"{m}|{t}|{g}")
                if not d:
                    continue
                rows.append({"spacecraft": g, "method": LABELS[m], "seeds": d["n_seeds"],
                             "event P": fmt(d, "event_precision"), "event R": fmt(d, "event_recall"),
                             "event F1": fmt(d, "event_f1"),
                             "point F1 (no adjust)": fmt(d, "point_f1"),
                             "PA-F1 (inflated)": fmt(d, "pa_f1"),
                             "false alarms / 1k steps": fmt(d, "fa_per_1k", 2)})
        df = pd.DataFrame(rows)
        df.to_csv(f"results/tables/detection_{t}.csv", index=False)
        Path(f"results/tables/detection_{t}.md").write_text(
            df.to_markdown(index=False) + f"\n\nThreshold: {t}, fitted on the validation split (end of train) only. "
            "Event-wise = telemanom definition; PA-F1 shown only for comparison with random scores.\n",
            encoding="utf-8")
        print(f"\n## threshold = {t}\n" + df.to_markdown(index=False))


def sanity(recs):
    """Which labeled anomalies do trivial baselines already catch (percentile threshold)?"""
    out = []
    for g in GROUPS:
        rs = [r for r in recs if g == "ALL" or r["spacecraft"] == g]
        tot = sum(len(r["label_seqs"]) for r in rs)
        row = {"spacecraft": g, "labeled events": tot}
        for m in ("persist", "zscore", "random", "tiny_int8", "lstm"):
            if m not in rs[0]["methods"]:
                continue
            for t in THRS:
                if t not in rs[0]["methods"][m]["0"]["thr"]:
                    continue
                tp = [sum(r["methods"][m][s]["thr"][t]["counts"]["event"][0] for r in rs)
                      for s in sorted(rs[0]["methods"][m])]
                row[f"{LABELS[m]} [{t}]"] = f"{np.mean(tp) / tot * 100:.1f}%"
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv("results/tables/benchmark_sanity.csv", index=False)
    Path("results/tables/benchmark_sanity.md").write_text(
        df.to_markdown(index=False) + "\n\nFraction of labeled anomaly sequences with at least one alarm "
        "(event recall), mean over seeds. False alarms are NOT reflected here; see detection tables.\n",
        encoding="utf-8")
    print("\n## benchmark sanity\n" + df.to_markdown(index=False))


def figures(S):
    plt = setup()
    ms = [m for m in METHODS if f"{m}|percentile|ALL" in S]
    nan = {"mean": np.nan, "std": 0.0}
    get = lambda m, t, g, k: S.get(f"{m}|{t}|{g}", {}).get(k, nan)
    fig, axs = plt.subplots(1, 3, figsize=(DBL_W, 2.5), sharey=True)
    thr = [("telemanom", "telemanom pipeline", 1.0), ("percentile", "fixed 99.5th pct.", 0.55),
           ("ndt", "NDT on validation", 0.25)]
    for ax, g in zip(axs, GROUPS):
        x = np.arange(len(ms))
        for j, (t, lab, alpha) in enumerate(thr):
            mu = [get(m, t, g, "event_f1")["mean"] for m in ms]
            sd = [get(m, t, g, "event_f1")["std"] for m in ms]
            ax.bar(x + (j - 1) * 0.27, mu, 0.27, yerr=sd, capsize=1.5, color=[COLORS[m] for m in ms],
                   alpha=alpha, label=lab)
        ax.set_xticks(x, [LABELS[m] for m in ms], rotation=45, ha="right")
        ax.set_title(g)
    axs[0].set_ylabel("event-wise F1 (mean ± sd over seeds)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend([x[0] if hasattr(x, "__getitem__") else x for x in h], l, loc="lower center", ncol=3, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, -0.3))
    save(fig, "figures/detection_f1")

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    t, g = "percentile", "ALL"
    x = np.arange(len(ms))
    for j, (k, lab) in enumerate([("point_f1", "point F1 (no adjust)"), ("event_f1", "event F1"),
                                  ("pa_f1", "point-adjusted F1")]):
        ax.bar(x + (j - 1) * 0.27, [get(m, t, g, k)["mean"] for m in ms], 0.27, label=lab)
    ax.set_xticks(x, [LABELS[m] for m in ms], rotation=45, ha="right")
    ax.set_ylabel("F1, SMAP + MSL (fixed threshold)")
    ax.legend(fontsize=6)
    ax.set_title("Point adjustment rewards random scores most")
    save(fig, "figures/pa_inflation")


def main():
    recs = load()
    S = summarize(recs)
    Path("results/detection_summary.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
    tables(S)
    sanity(recs)
    figures(S)


if __name__ == "__main__":
    main()
