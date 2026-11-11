import numpy as np
import torch

from src.faults.inject import flip_bits, flip_int32_bit
from src.faults.protect import checksums, crc32, sanity_filter, scrub, tmr_vote
from src.models.nets import TinyMLP
from src.quant.int8 import fake_quant_forward, float_forward, float_layers, quantize, requant, \
    quantize_multiplier


def _model(seed=0, in_dim=88):
    torch.manual_seed(seed)
    m = TinyMLP(in_dim)
    X = np.random.default_rng(seed).normal(0, 1, (2000, in_dim)).astype(np.float32)
    return m, X


def test_requant_matches_float_rounding():
    rng = np.random.default_rng(0)
    for m in [0.3, 1e-3, 0.77, 2.5]:
        M0, sh = quantize_multiplier(np.array([m]))
        acc = rng.integers(-2 ** 20, 2 ** 20, 5000)
        ref = np.round(acc * m)
        got = requant(acc[:, None], M0, sh)[:, 0]
        assert np.max(np.abs(got - ref)) <= 1


def test_integer_model_matches_fake_quant():
    m, X = _model()
    layers = float_layers(m)
    qm = quantize(layers, X)
    Xt = np.random.default_rng(5).normal(0, 1, (3000, X.shape[1])).astype(np.float32)
    yi = qm.forward_int(qm.quantize_input(Xt)).astype(np.int64)
    yf = np.round(fake_quant_forward(qm, Xt) / qm.s_out).astype(np.int64)
    # integer pipeline vs float simulation of the same arithmetic: within 1 LSB
    assert np.max(np.abs(yi - yf)) <= 1
    assert np.mean(yi == yf) > 0.95
    # and close to the float model in real units
    err = np.abs(qm.predict(Xt) - float_forward(layers, Xt))
    assert np.median(err) < 0.05 * np.abs(float_forward(layers, Xt)).max()


def test_flip_twice_restores_and_golden_unchanged():
    m, X = _model()
    qm = quantize(float_layers(m), X)
    gold = [w.copy() for w in qm.W]
    rng = np.random.default_rng(1)
    new, flips = flip_bits(qm.W, 20, rng)
    assert len(flips) == 20 and len(set(flips)) == 20
    assert all(np.array_equal(a, b) for a, b in zip(qm.W, gold))     # golden untouched
    nbits = sum(np.unpackbits((a.view(np.uint8) ^ b.view(np.uint8)).reshape(-1)).sum()
                for a, b in zip(new, gold))
    assert nbits == 20
    # flipping the same bits again restores the original
    back = [a.copy() for a in new]
    for ai, wi, bi in flips:
        u = back[ai].view(np.uint8).reshape(-1)
        u[wi * back[ai].itemsize + bi // 8] ^= np.uint8(1 << (bi % 8))
    assert all(np.array_equal(a, b) for a, b in zip(back, gold))


def test_bit_position_restriction():
    a = [np.zeros(1000, np.int8)]
    new, flips = flip_bits(a, 50, np.random.default_rng(2), bit_positions=[7])
    assert all(b == 7 for _, _, b in flips)
    assert np.all((new[0] == 0) | (new[0] == -128))
    f = [np.ones(500, np.float32)]
    newf, flf = flip_bits(f, 30, np.random.default_rng(3), bit_positions=range(23, 31))
    assert all(23 <= b <= 30 for _, _, b in flf)


def test_checksum_detects_every_single_bit_flip_and_scrub_restores():
    a = np.random.default_rng(4).integers(-127, 128, 300).astype(np.int8)
    s = crc32(a)
    for bit in range(a.size * 8):
        c = a.copy()
        c.view(np.uint8)[bit // 8] ^= np.uint8(1 << (bit % 8))
        assert crc32(c) != s
    bad = a.copy()
    bad[17] ^= 4
    fixed, idx = scrub([bad], [a], checksums([a]))
    assert idx == [0] and np.array_equal(fixed[0], a)


def test_tmr_majority_masks_single_copy_faults():
    a = np.random.default_rng(6).integers(-127, 128, 200).astype(np.int8)
    b, c = a.copy(), a.copy()
    b[3] ^= 0x40
    c[9] ^= 0x01
    assert np.array_equal(tmr_vote(a, b, c), a)


def test_int32_flip():
    assert flip_int32_bit(np.array([0]), 31)[0] == -(1 << 31)
    assert flip_int32_bit(np.array([5]), 0)[0] == 4


def test_sanity_filter():
    out, n = sanity_filter(np.array([0.5, 10.0, np.nan]), np.array([0.1, 0.2, 0.3]), 0.0, 1.0)
    assert n == 2 and np.allclose(out, [0.5, 0.2, 0.3])
