import inspect

import numpy as np

from src.detect.metrics import (aggregate, all_counts, event_counts, mask_to_seqs, point_adjust,
                                point_counts, seqs_to_mask)
from src.detect.threshold import ewma, ndt_threshold, percentile_threshold


def test_mask_seq_roundtrip():
    seqs = [(2, 4), (7, 7), (9, 11)]
    assert mask_to_seqs(seqs_to_mask(seqs, 12)) == seqs


def test_event_counts_known_case():
    labels = [(5, 9), (20, 24)]
    pred = np.zeros(30, bool)
    pred[7] = True          # hits first event
    pred[12:14] = True      # one FP event
    pred[16] = True         # second FP event
    assert event_counts(pred, labels) == (1, 2, 1)


def test_point_counts_and_pa():
    labels = [(0, 9)]
    pred = np.zeros(20, bool)
    pred[3] = True
    pred[15] = True
    assert point_counts(pred, labels) == (1, 1, 9)
    pa = point_adjust(pred, labels)
    assert pa[:10].all() and pa[15] and pa.sum() == 11
    assert point_counts(pa, labels) == (10, 1, 0)


def test_fp_run_touching_label_is_not_fp():
    labels = [(5, 9)]
    pred = np.zeros(15, bool)
    pred[8:13] = True       # run overlaps the label -> TP, no FP event
    assert event_counts(pred, labels) == (1, 0, 0)


def test_random_scores_pa_inflation():
    """Kim et al. 2022: random scores + point adjustment give high F1 on long events."""
    rng = np.random.default_rng(0)
    n = 10000
    labels = [(s, s + 299) for s in range(500, 9000, 1500)]
    pred = rng.random(n) > 0.99           # 1% random alarms
    c = aggregate([all_counts(pred, labels)])
    assert c["pa"]["f1"] > 0.5
    assert c["point"]["f1"] < 0.1
    assert c["pa"]["f1"] > 5 * c["point"]["f1"]


def test_thresholds_take_no_labels():
    for f in (percentile_threshold, ndt_threshold):
        params = inspect.signature(f).parameters
        assert not any("label" in p or "seq" in p for p in params)


def test_ndt_separates_obvious_outliers():
    rng = np.random.default_rng(1)
    e = np.abs(rng.normal(0, 1, 5000))
    e[1000:1010] += 20
    eps = ndt_threshold(e)
    assert e[1000:1010].min() > eps and (e > eps).sum() < 50


def test_ewma_matches_loop():
    x = np.random.default_rng(2).random(200)
    acc, ref = x[0], []
    for v in x:
        acc = 0.1 * v + 0.9 * acc
        ref.append(acc)
    assert np.allclose(ewma(x, 0.1), ref)
