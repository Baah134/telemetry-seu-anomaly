"""NASA SMAP/MSL telemetry (Hundman et al., KDD 2018, telemanom release).

Layout expected (verified on load): data/raw/train/<chan>.npy, data/raw/test/<chan>.npy,
data/raw/labeled_anomalies.csv. Column 0 is the telemetry value; remaining columns are
one-hot command features. Anomaly sequences are [start, end] indices into the test array
(inclusive, as in telemanom).

Windowing: input at time t = [y[t-W], ..., y[t-1]] plus the command vector at t-1;
target y[t]. The first W test points have no forecast and are never flagged (the same
for every method).
Validation split: the last VAL_FRAC of each TRAIN series (chronological, never random).
Normalization: per-channel mean/std of the train (minus validation) part only.
"""
import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import os
RAW = Path(os.environ.get("TELEM_DATA_RAW", ROOT / "data" / "raw"))
W = 64
VAL_FRAC = 0.2


@dataclass
class Channel:
    chan_id: str
    spacecraft: str
    train: np.ndarray        # (n_train, 1 + n_cmd), raw
    test: np.ndarray         # (n_test, 1 + n_cmd), raw
    label_seqs: list         # [(start, end_inclusive)] in test indices
    classes: list
    mu: float = 0.0
    sd: float = 1.0

    @property
    def n_cmd(self):
        return self.train.shape[1] - 1

    def split(self):
        n_val = int(round(VAL_FRAC * len(self.train)))
        return self.train[:-n_val], self.train[-n_val:]


def merge_seqs(seqs):
    """Union of inclusive [start, end] sequences, overlapping/adjacent ones merged."""
    out = []
    for s, e in sorted((int(a), int(b)) for a, b in seqs):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def load_labels(path=RAW / "labeled_anomalies.csv"):
    """One row per channel. The official CSV lists P-2 twice with overlapping labels
    ([5350, 6575] and [5300, 6420]); duplicate rows are merged (label union)."""
    df = pd.read_csv(path)
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)
    rows = []
    for cid, g in df.groupby("chan_id", sort=False):
        r = g.iloc[0].to_dict()
        if len(g) > 1:
            if g.spacecraft.nunique() != 1 or g.num_values.nunique() != 1:
                raise ValueError(f"{cid}: duplicate rows disagree on spacecraft/num_values")
            r["anomaly_sequences"] = merge_seqs([s for seqs in g.anomaly_sequences for s in seqs])
            r["class"] = "[" + ", ".join(str(c).strip("[]") for c in g["class"]) + "]"
            r["n_label_rows"] = len(g)
        rows.append(r)
    return pd.DataFrame(rows)


def load_channel(row, raw=RAW):
    row = dict(row)
    tr = np.load(raw / "train" / f"{row['chan_id']}.npy").astype(np.float32)
    te = np.load(raw / "test" / f"{row['chan_id']}.npy").astype(np.float32)
    if tr.ndim != 2 or te.ndim != 2 or tr.shape[1] != te.shape[1]:
        raise ValueError(f"{row['chan_id']}: unexpected shapes {tr.shape} {te.shape}")
    if len(te) != int(row["num_values"]):
        raise ValueError(f"{row['chan_id']}: test length {len(te)} != num_values {row['num_values']}")
    seqs = [(int(s), int(e)) for s, e in row["anomaly_sequences"]]
    for s, e in seqs:
        if not (0 <= s <= e < len(te)):
            raise ValueError(f"{row['chan_id']}: label {s}-{e} outside test range {len(te)}")
    ch = Channel(row["chan_id"], row["spacecraft"], tr, te, seqs,
                 [c.strip() for c in str(row["class"]).strip("[]").split(",")])
    fit, _ = ch.split()
    ch.mu = float(fit[:, 0].mean())
    ch.sd = float(fit[:, 0].std()) or 1.0
    return ch


def load_all(raw=RAW, spacecraft=None):
    df = load_labels(raw / "labeled_anomalies.csv")
    if spacecraft:
        df = df[df.spacecraft == spacecraft]
    # telemanom lists each channel once; keep order stable
    return [load_channel(r, raw) for r in df.to_dict("records")]


def windows(arr, ch, w=W):
    """arr (n, 1+n_cmd) raw -> X (n-w, w + n_cmd) normalized, y (n-w,) normalized."""
    v = (arr[:, 0] - ch.mu) / ch.sd
    n = len(arr)
    if n <= w:
        return np.zeros((0, w + ch.n_cmd), np.float32), np.zeros(0, np.float32)
    idx = np.arange(w)[None, :] + np.arange(n - w)[:, None]
    X = np.concatenate([v[idx], arr[w - 1:n - 1, 1:]], axis=1).astype(np.float32)
    return X, v[w:].astype(np.float32)


def full_series_pred(pred_tail, n, w=W, fill=np.nan):
    """Place forecasts for t = w..n-1 into a length-n array."""
    out = np.full(n, fill, dtype=np.float64)
    out[w:] = pred_tail
    return out
