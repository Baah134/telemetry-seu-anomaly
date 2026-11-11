"""Post-training INT8 quantization and an integer-only NumPy reference (TFLite Micro style).

Scheme
- Weights: symmetric per-output-channel int8, s_w[j] = max|W[j,:]| / 127, zero point 0.
- Activations (input, hidden, output): symmetric per-tensor int8, s_a = max|a| / 127
  profiled on the TRAINING windows (calibration), zero point 0.
- Bias: int32, b_q[j] = round(b[j] / (s_in * s_w[j])).
- Layer: acc_j = sum_i W_q[j,i] * x_q[i] + b_q[j]  (int32 accumulator)
         y_q[j] = sat_int8( round(acc_j * M_j) ),  M_j = s_in * s_w[j] / s_out
  M_j is applied as a fixed-point multiplier M0 (int32, Q31) and a right shift, with the
  rounding-doubling high multiply and rounding right shift used by TFLite / gemmlowp.
- ReLU on hidden layers = clamp the int8 output to [0, 127] (zero point is 0).

`IntModel` holds every stored value as a NumPy integer array so that fault injection
flips bits in exactly the bytes an MCU would hold.
"""
from dataclasses import dataclass, field

import numpy as np

QMAX = 127


def quantize_multiplier(m):
    """real multiplier m in (0, 1) -> (M0 int32 in [2^30, 2^31), shift >= 0) with m ~ M0 * 2^-31 * 2^-shift."""
    m = np.asarray(m, dtype=np.float64)
    M0 = np.zeros(m.shape, np.int64)
    sh = np.zeros(m.shape, np.int64)
    for i, v in np.ndenumerate(m):
        if v <= 0:
            continue
        mant, exp = np.frexp(v)                 # v = mant * 2^exp, mant in [0.5, 1)
        q = int(round(mant * (1 << 31)))
        if q == (1 << 31):
            q //= 2
            exp += 1
        M0[i], sh[i] = q, -exp
    return M0, sh


def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul(a int32, b int32) on int64 arrays."""
    a = a.astype(np.int64)
    b = np.asarray(b, np.int64)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    # Python // floors; gemmlowp truncates toward zero -> emulate
    num = ab + nudge
    q = np.where(num >= 0, num >> 31, -((-num) >> 31))
    return np.clip(q, -(1 << 31), (1 << 31) - 1)


def rounding_rshift(x, sh):
    sh = np.asarray(sh, np.int64)
    mask = (np.int64(1) << sh) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0).astype(np.int64)
    return (x >> sh) + (rem > thr).astype(np.int64)


def requant(acc, M0, sh):
    """sh < 0 means multiplier >= 1: left-shift first (TFLite does the same)."""
    sh = np.asarray(sh, np.int64)
    left = np.maximum(-sh, 0)
    acc = np.clip(acc.astype(np.int64) << left, -(1 << 31), (1 << 31) - 1)
    return rounding_rshift(srdhm(acc, M0), np.maximum(sh, 0))


@dataclass
class IntModel:
    W: list            # int8 arrays (out, in)
    b: list            # int32 arrays (out,)
    M0: list           # int32 multipliers per layer (out,)
    sh: list           # shifts per layer (out,)
    s_in: float        # input scale
    s_out: float       # output scale (last layer)
    s_act: list = field(default_factory=list)   # activation scales [in, h1, h2, out]
    s_w: list = field(default_factory=list)

    def copy(self):
        return IntModel([w.copy() for w in self.W], [b.copy() for b in self.b], list(self.M0),
                        list(self.sh), self.s_in, self.s_out, list(self.s_act), list(self.s_w))

    @property
    def n_weight_bytes(self):
        return sum(w.size for w in self.W) + 4 * sum(b.size for b in self.b)

    def quantize_input(self, X):
        return np.clip(np.round(X / self.s_in), -QMAX, QMAX).astype(np.int8)

    def forward_int(self, xq, act_hook=None, act_clamp=None):
        """xq int8 (N, in) -> int8 output codes (N,). act_hook(layer, acc) may corrupt the
        int32 accumulators in place (transient faults). act_clamp: per-layer int8 upper
        bounds (range restriction)."""
        a = xq.astype(np.int64)
        L = len(self.W)
        for l in range(L):
            # exact integer GEMV via float64 BLAS: |sum| <= 127*127*in + 2^31 << 2^53
            acc = (a.astype(np.float64) @ self.W[l].astype(np.float64).T).astype(np.int64) \
                + self.b[l].astype(np.int64)
            acc = np.clip(acc, -(1 << 31), (1 << 31) - 1)          # int32 accumulator
            if act_hook is not None:
                acc = act_hook(l, acc)
            y = requant(acc, self.M0[l], self.sh[l])
            lo = 0 if l < L - 1 else -QMAX                          # ReLU on hidden layers
            y = np.clip(y, lo, QMAX)
            if act_clamp is not None and act_clamp[l] is not None:
                y = np.clip(y, act_clamp[l][0], act_clamp[l][1])
            a = y
        return a[:, 0].astype(np.int8)

    def predict(self, X, **kw):
        return self.forward_int(self.quantize_input(X), **kw).astype(np.float64) * self.s_out


def float_layers(model):
    """TinyMLP (torch) -> [(W float32 (out,in), b float32)]."""
    return [(m.weight.detach().numpy().astype(np.float32), m.bias.detach().numpy().astype(np.float32))
            for m in (model.fc1, model.fc2, model.fc3)]


def float_forward(layers, X, act_clamp=None):
    a = X.astype(np.float32)
    for l, (W, b) in enumerate(layers):
        a = a @ W.T + b
        if l < len(layers) - 1:
            a = np.maximum(a, 0)
        if act_clamp is not None and act_clamp[l] is not None:
            a = np.clip(a, act_clamp[l][0], act_clamp[l][1])
    return a[:, 0]


def activation_ranges(layers, X):
    """max |activation| per layer output, from calibration data (training windows)."""
    a = X.astype(np.float32)
    out = [float(np.abs(a).max())]
    for l, (W, b) in enumerate(layers):
        a = a @ W.T + b
        if l < len(layers) - 1:
            a = np.maximum(a, 0)
        out.append(float(np.abs(a).max()))
    return out   # [input, h1, h2, out]


def quantize(layers, X_calib):
    r = activation_ranges(layers, X_calib)
    s_act = [max(v, 1e-8) / QMAX for v in r]
    Wq, bq, M0s, shs, s_ws = [], [], [], [], []
    for l, (W, b) in enumerate(layers):
        s_w = np.maximum(np.abs(W).max(axis=1), 1e-12) / QMAX
        Wq.append(np.clip(np.round(W / s_w[:, None]), -QMAX, QMAX).astype(np.int8))
        bq.append(np.round(b / (s_act[l] * s_w)).astype(np.int32))
        M0, sh = quantize_multiplier(s_act[l] * s_w / s_act[l + 1])
        M0s.append(M0)
        shs.append(sh)
        s_ws.append(s_w)
    return IntModel(Wq, bq, M0s, shs, s_act[0], s_act[-1], s_act, s_ws)


def fake_quant_forward(qm, X):
    """Float simulation of the same quantized arithmetic (dequantized weights, activations
    rounded to their int8 grid after each layer). Reference for the integer model."""
    a = np.clip(np.round(X / qm.s_in), -QMAX, QMAX) * qm.s_in
    L = len(qm.W)
    for l in range(L):
        Wf = qm.W[l].astype(np.float64) * qm.s_w[l][:, None]
        bf = qm.b[l].astype(np.float64) * qm.s_act[l] * qm.s_w[l]
        z = a @ Wf.T + bf
        s = qm.s_act[l + 1]
        lo = 0 if l < L - 1 else -QMAX
        a = np.clip(np.round(z / s), lo, QMAX) * s
    return a[:, 0]
