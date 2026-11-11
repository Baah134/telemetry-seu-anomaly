"""Anomaly-detection metrics. All functions take boolean arrays over test timesteps
and lists of labeled anomaly sequences [(start, end_inclusive), ...].

- event-wise (telemanom style, Hundman et al. 2018): a labeled sequence is a TP if any
  predicted-anomalous point overlaps it; FN otherwise. Each contiguous run of predicted
  points that overlaps no labeled sequence is ONE false-positive event.
- point-wise: plain per-timestep precision/recall/F1, no adjustment.
- point-adjusted (PA): if any point inside a labeled sequence is predicted, the whole
  sequence counts as predicted (Xu et al. 2018 style). Inflates scores; reported only
  next to a random-score baseline (Kim et al., AAAI 2022).
"""
import numpy as np


def seqs_to_mask(seqs, n):
    m = np.zeros(n, dtype=bool)
    for s, e in seqs:
        m[max(0, s):min(n, e + 1)] = True
    return m


def mask_to_seqs(mask):
    """Contiguous True runs -> [(start, end_inclusive)]."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def event_counts(pred, label_seqs):
    pred = np.asarray(pred, dtype=bool)
    lab = seqs_to_mask(label_seqs, pred.size)
    tp = sum(bool(pred[max(0, s):e + 1].any()) for s, e in label_seqs)
    fn = len(label_seqs) - tp
    fp = sum(1 for s, e in mask_to_seqs(pred) if not lab[s:e + 1].any())
    return tp, fp, fn


def point_counts(pred, label_seqs):
    pred = np.asarray(pred, dtype=bool)
    lab = seqs_to_mask(label_seqs, pred.size)
    return int(np.sum(pred & lab)), int(np.sum(pred & ~lab)), int(np.sum(~pred & lab))


def point_adjust(pred, label_seqs):
    pa = np.asarray(pred, dtype=bool).copy()
    for s, e in label_seqs:
        if pa[s:e + 1].any():
            pa[s:e + 1] = True
    return pa


def all_counts(pred, label_seqs):
    """Raw counts for one channel; aggregate by summing counts, then computing P/R/F1."""
    return {
        "event": event_counts(pred, label_seqs),
        "point": point_counts(pred, label_seqs),
        "pa": point_counts(point_adjust(pred, label_seqs), label_seqs),
        "n": int(np.asarray(pred).size),
    }


def aggregate(counts_list):
    """Micro-aggregate over channels: sum TP/FP/FN, then P/R/F1."""
    out = {}
    for k in ("event", "point", "pa"):
        tp = sum(c[k][0] for c in counts_list)
        fp = sum(c[k][1] for c in counts_list)
        fn = sum(c[k][2] for c in counts_list)
        p, r, f = prf(tp, fp, fn)
        out[k] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}
    n = sum(c["n"] for c in counts_list)
    out["false_alarms_per_1k"] = 1000.0 * out["event"]["fp"] / max(1, n)
    return out
