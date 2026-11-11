"""Lightweight, MCU-friendly protections against weight/activation upsets.

1. range restriction: clamp each layer's output to bounds profiled on training data.
   int8 model: per-layer [lo, hi] in int8 codes; float model: [lo, hi] in float.
2. checksum + scrubbing: CRC32 per stored array (4 bytes each), checked every K
   inferences; on mismatch reload that array from a golden copy (assumed in ROM /
   protected memory). Between an upset and the next check the corrupted weights are used.
3. selective TMR: chosen arrays are stored three times; reads use the bitwise majority
   of the three copies. Faults are injected uniformly over ALL stored bits, including the
   two extra copies, so the extra memory is also exposed to upsets (fair accounting).
4. output sanity check: a forecast outside [min, max] of the training targets, widened
   by a margin of `margin` x range, is replaced by the last observed value (persistence).
"""
import zlib

import numpy as np


def crc32(a):
    return zlib.crc32(np.ascontiguousarray(a).view(np.uint8).tobytes()) & 0xFFFFFFFF


def checksums(arrs):
    return [crc32(a) for a in arrs]


def scrub(arrs, golden, sums):
    """Return (repaired arrays, list of indices that failed the check)."""
    out, bad = [], []
    for i, (a, g, s) in enumerate(zip(arrs, golden, sums)):
        if crc32(a) != s:
            out.append(g.copy())
            bad.append(i)
        else:
            out.append(a)
    return out, bad


def tmr_vote(a, b, c):
    """Bitwise majority of three same-dtype arrays."""
    ua, ub, uc = (np.ascontiguousarray(x).view(np.uint8) for x in (a, b, c))
    v = (ua & ub) | (ua & uc) | (ub & uc)
    return v.view(a.dtype).reshape(a.shape)


def sanity_filter(pred, last_obs, lo, hi, margin=0.1):
    """Replace implausible forecasts by persistence. Returns (filtered, n_replaced)."""
    span = hi - lo
    bad = (pred < lo - margin * span) | (pred > hi + margin * span) | ~np.isfinite(pred)
    out = np.where(bad, last_obs, pred)
    return out, int(bad.sum())
