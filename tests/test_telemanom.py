"""Behavioural tests for the telemanom port (known-answer cases)."""
import numpy as np

from src.data.loader import merge_seqs
from src.detect.telemanom import detect, smooth


def _base(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.sin(np.arange(n) / 50.0)
    e = np.abs(0.02 * rng.normal(size=n))
    return y, e


def test_clear_error_burst_is_detected_with_buffer():
    y, e = _base()
    e[3000:3050] += 1.0
    seqs = detect(e, y)
    assert len(seqs) == 1
    s, t = seqs[0]
    assert s <= 3000 and t >= 3049          # covers the burst (plus error buffer)
    assert 3000 - s <= 200 and t - 3049 <= 300


def test_quiet_errors_give_no_alarm():
    y, e = _base(seed=1)
    assert detect(e, y) == []


def test_tiny_errors_below_absolute_floor_are_ignored():
    # telemanom skips windows whose max smoothed error is <= 0.05 (absolute units)
    y, e = _base(seed=2)
    e = e * 0.1
    e[3000:3050] += 0.04
    assert detect(e, y) == []


def test_smoothing_span_and_startup_fix():
    e = np.zeros(1000)
    e[500] = 1.0
    s = smooth(e, l_s=64)
    assert np.allclose(s[:64], np.mean(s[:128]))
    assert s[500] > s[499] and s[600] < s[501]


def test_merge_seqs_for_duplicate_label_rows():
    assert merge_seqs([[5350, 6575], [5300, 6420]]) == [(5300, 6575)]
    assert merge_seqs([[1, 5], [10, 12], [6, 8]]) == [(1, 8), (10, 12)]


def test_dilate_matches_telemanom_buffer_expansion():
    from src.detect.telemanom import _dilate
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(50, 3000))
        idx = np.unique(rng.integers(0, n, int(rng.integers(1, 40))))
        b = int(rng.integers(1, 120))
        buf = np.arange(1, b)
        ref = np.unique(np.concatenate([idx, (idx[:, None] + buf).ravel(), (idx[:, None] - buf).ravel()]))
        ref = ref[(ref >= 0) & (ref < n)]
        assert np.array_equal(_dilate(idx, n, b), ref)
