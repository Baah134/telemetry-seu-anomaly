"""Inspect the SMAP/MSL data: channel counts, lengths, anomaly counts and fractions,
example plots. Writes results/data_summary.json, results/tables/data_channels.csv,
figures/data_examples.{png,pdf}. Usage: python -m scripts.inspect_data
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.style import COL_W, DBL_W, save, setup
from src.data.loader import load_all
from src.detect.metrics import seqs_to_mask


def main():
    chans = load_all()
    rows = []
    for c in chans:
        m = seqs_to_mask(c.label_seqs, len(c.test))
        rows.append({"chan_id": c.chan_id, "spacecraft": c.spacecraft, "n_train": len(c.train),
                     "n_test": len(c.test), "n_cmd": c.n_cmd, "n_anomalies": len(c.label_seqs),
                     "anomaly_frac": float(m.mean()),
                     "mean_anomaly_len": float(np.mean([e - s + 1 for s, e in c.label_seqs])),
                     "classes": ";".join(c.classes),
                     "train_min": float(c.train[:, 0].min()), "train_max": float(c.train[:, 0].max()),
                     "test_min": float(c.test[:, 0].min()), "test_max": float(c.test[:, 0].max())})
    df = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/data_channels.csv", index=False)
    summ = {}
    for sc, g in list(df.groupby("spacecraft")) + [("ALL", df)]:
        summ[sc] = {"channels": int(len(g)), "anomaly_sequences": int(g.n_anomalies.sum()),
                    "test_points": int(g.n_test.sum()), "train_points": int(g.n_train.sum()),
                    "anomaly_point_frac": float((g.anomaly_frac * g.n_test).sum() / g.n_test.sum()),
                    "n_cmd": sorted(set(int(x) for x in g.n_cmd)),
                    "median_anomaly_len": float(g.mean_anomaly_len.median()),
                    "max_channel_anomaly_frac": float(g.anomaly_frac.max()),
                    "point_anomalies": int(sum(cl.count("point") for cl in g.classes)),
                    "contextual_anomalies": int(sum(cl.count("contextual") for cl in g.classes))}
    Path("results/data_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))

    plt = setup()
    ex = [c for c in chans if c.chan_id in ("P-1", "T-1", "C-1", "M-6")][:4] or chans[:4]
    fig, axs = plt.subplots(len(ex), 1, figsize=(DBL_W, 1.3 * len(ex)))
    for ax, c in zip(np.atleast_1d(axs), ex):
        ax.plot(c.test[:, 0], lw=0.5, color="k")
        for s, e in c.label_seqs:
            ax.axvspan(s, e, color="tab:red", alpha=0.25, lw=0)
        ax.set_ylabel(f"{c.chan_id}\n({c.spacecraft})")
    np.atleast_1d(axs)[-1].set_xlabel("test index (labeled anomalies shaded)")
    save(fig, "figures/data_examples")


if __name__ == "__main__":
    main()
