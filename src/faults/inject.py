"""Simulated single-event upsets (bit flips). Simulation only, not radiation testing.

Weight memory = every stored parameter word: int8 weights (8 bits each) and int32 biases
(32 bits each) for the int8 model; float32 weights and biases (32 bits each) for the
float model. A fault = flipping k distinct bits chosen uniformly over all bits of the
targeted memory (optionally restricted to one layer / one bit position).
Golden copies are never modified: injection always returns new arrays.
"""
import numpy as np


def _views(arrs):
    """List of arrays -> list of (uint8 byte view of a COPY, dtype, shape)."""
    out = []
    for a in arrs:
        c = np.ascontiguousarray(a).copy()
        out.append(c)
    return out


def memory_map(arrs):
    """Bit counts per array; returns cumulative offsets for uniform bit sampling."""
    bits = np.array([a.size * a.itemsize * 8 for a in arrs], dtype=np.int64)
    return bits, np.concatenate([[0], np.cumsum(bits)])


def flip_bits(arrs, k, rng, bit_positions=None, word_bits_filter=None):
    """Flip k distinct bits across the list of arrays. Returns (new_arrays, flips) where
    flips = [(array_index, flat_word_index, bit_in_word)].

    bit_positions: optional iterable of allowed bit indices within a word (0 = LSB); used
    for bit-position sensitivity (e.g. [7] = int8 sign bit, [23..30] = float32 exponent).
    """
    new = _views(arrs)
    words = np.array([a.size for a in new], dtype=np.int64)
    wbits = np.array([a.itemsize * 8 for a in new], dtype=np.int64)
    if bit_positions is None:
        slots = words * wbits
        total = int(slots.sum())
        if k > total:
            raise ValueError("more flips than bits")
        picks = rng.choice(total, size=k, replace=False)
        off = np.concatenate([[0], np.cumsum(slots)])
        flips = []
        for p in picks:
            ai = int(np.searchsorted(off, p, side="right") - 1)
            r = int(p - off[ai])
            flips.append((ai, r // int(wbits[ai]), r % int(wbits[ai])))
    else:
        bp = np.asarray(sorted(set(bit_positions)), dtype=np.int64)
        total = int(words.sum()) * len(bp)
        picks = rng.choice(total, size=min(k, total), replace=False)
        off = np.concatenate([[0], np.cumsum(words * len(bp))])
        flips = []
        for p in picks:
            ai = int(np.searchsorted(off, p, side="right") - 1)
            r = int(p - off[ai])
            flips.append((ai, r // len(bp), int(bp[r % len(bp)])))
    for ai, wi, bi in flips:
        a = new[ai]
        u = a.view(np.uint8).reshape(-1)
        byte = wi * a.itemsize + bi // 8          # little-endian word layout
        u[byte] ^= np.uint8(1 << (bi % 8))
    return new, flips


def flip_int32_bit(x, bit):
    """Flip one bit of int32 values (array), returns int64 array of the result."""
    v = (np.asarray(x, np.int64) & 0xFFFFFFFF) ^ (1 << bit)
    return np.where(v >= (1 << 31), v - (1 << 32), v)
