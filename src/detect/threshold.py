"""Scoring and thresholding. Threshold functions NEVER receive labels: their
signatures take only error arrays, so test labels cannot leak in.

score:      EWMA-smoothed absolute forecast error (telemanom smooths errors with EWMA).
percentile: fixed threshold = q-th percentile of smoothed errors on the validation
            split (chronological end of the training series).
ndt:        simplified nonparametric dynamic threshold (Hundman et al. 2018, Sec. 3.2):
            for z in Z, eps = mu + z*sigma; choose eps maximizing
              (dmu/mu + dsigma/sigma) / (n_above + n_seq^2)
            where dmu, dsigma are the drops in mean/std of the errors after removing
            values above eps, n_above their count and n_seq the number of contiguous
            runs above eps. Telemanom computes this on sliding windows of TEST errors
            (label-free but test-adaptive). Here it is computed once on validation
            errors and applied as a fixed threshold. See docs/DECISIONS.md.
"""
import numpy as np

from .metrics import mask_to_seqs


def ewma(x, alpha=0.05):
    """y[t] = alpha*x[t] + (1-alpha)*y[t-1], y[-1] = x[0]. Vectorized via lfilter."""
    from scipy.signal import lfilter
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    zi = np.array([(1 - alpha) * x[0]])
    y, _ = lfilter([alpha], [1, -(1 - alpha)], x, zi=zi)
    return y


def smoothed_error(y_true, y_pred, alpha=0.05):
    return ewma(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)), alpha)


def percentile_threshold(val_scores, q=99.5):
    return float(np.percentile(val_scores, q))


def ndt_threshold(val_scores, z_range=np.arange(2.0, 10.5, 0.5)):
    e = np.asarray(val_scores, dtype=np.float64)
    mu, sd = e.mean(), e.std()
    best, best_eps = -np.inf, mu + z_range[-1] * sd
    for z in z_range:
        eps = mu + z * sd
        below = e[e < eps]
        above = e >= eps
        if not above.any() or below.size < 2:
            continue
        dmu = (mu - below.mean()) / mu if mu > 0 else 0.0
        dsd = (sd - below.std()) / sd if sd > 0 else 0.0
        nseq = len(mask_to_seqs(above))
        val = (dmu + dsd) / (above.sum() + nseq ** 2)
        if val > best:
            best, best_eps = val, eps
    return float(best_eps)


THRESHOLDS = {"percentile": percentile_threshold, "ndt": ndt_threshold}
