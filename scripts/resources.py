"""Resource accounting: parameters, weight bytes, peak activation memory, MACs per
inference, measured host latency, and ESTIMATED ESP32 latency/energy.

ESP32 estimate (not a measurement; Prince can measure with export/esp32_sketch):
  - clock 240 MHz (ESP32-WROOM-32 default max)
  - CYCLES_PER_MAC_INT8 = 2..6: scalar C loop on Xtensa LX6 with no SIMD (load, load,
    multiply-accumulate, loop overhead). ASSUMED range, not measured or cited.
  - CYCLES_PER_MAC_F32 = 4..10 (single-precision FPU, scalar). ASSUMED range.
  - requantization adds ~20 cycles per output neuron (two 64-bit multiplies, shifts).
  - power: ESP32 datasheet (Espressif, v4.x, "Power consumption by power modes"),
    modem-sleep, 240 MHz dual-core, CPU on and radio off: 30-68 mA at 3.3 V.
    Energy = latency x 3.3 V x current.
Writes results/resources.json and results/tables/resources.{csv,md}.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CLK = 240e6
CPM_I8 = (2, 6)
CPM_F32 = (4, 10)
REQ_CYC = 20
V, I_MA = 3.3, (30, 68)


def mlp_macs(in_dim, h1=32, h2=16):
    return in_dim * h1 + h1 * h2 + h2


def lstm_macs(w, n_cmd, h=80):
    l1 = 4 * (1 + n_cmd + h) * h
    l2 = 4 * (h + h) * h
    return w * (l1 + l2) + h


def lstm_params(n_cmd, h=80):
    return 4 * ((1 + n_cmd) * h + h * h + 2 * h) + 4 * (h * h + h * h + 2 * h) + h + 1


def est(macs, cpm, extra_cycles=0):
    lo = (macs * cpm[0] + extra_cycles) / CLK
    hi = (macs * cpm[1] + extra_cycles) / CLK
    return lo, hi


def main():
    from src.data.loader import W
    from src.models.nets import TinyMLP
    from src.quant.int8 import float_layers, quantize
    import torch
    rows = []
    for sc, n_cmd, n_ch in (("SMAP", 24, 54), ("MSL", 54, 27)):
        d = W + n_cmd
        m = TinyMLP(d)
        p = sum(x.numel() for x in m.parameters())
        X = np.random.default_rng(0).normal(size=(4096, d)).astype(np.float32)
        qm = quantize(float_layers(m), X)
        mac = mlp_macs(d)
        # host latency, single-window inference (batch 1), median of 2000
        x1 = X[:1]
        ts = []
        for _ in range(2000):
            t = time.perf_counter()
            qm.predict(x1)
            ts.append(time.perf_counter() - t)
        host_i8 = float(np.median(ts))
        with torch.no_grad():
            xt = torch.from_numpy(x1)
            ts = []
            for _ in range(2000):
                t = time.perf_counter()
                m(xt)
                ts.append(time.perf_counter() - t)
        host_f32 = float(np.median(ts))
        li8 = est(mac, CPM_I8, REQ_CYC * (32 + 16 + 1))
        lf = est(mac, CPM_F32)
        lm = lstm_macs(W, n_cmd)
        ll = est(lm, CPM_F32)
        lp = lstm_params(n_cmd)
        for name, params, wbytes, act, macs, host, lat in [
            ("Tiny MLP INT8 (per channel)", p, qm.n_weight_bytes, d + 32 + 16, mac, host_i8, li8),
            ("Tiny MLP float32 (per channel)", p, 4 * p, 4 * (d + 32 + 16), mac, host_f32, lf),
            ("LSTM 2x80 float32 (per channel)", lp, 4 * lp, 4 * (W * (1 + n_cmd) + 4 * 80 * 2 + 2 * 80), lm, None, ll),
        ]:
            rows.append({"spacecraft": sc, "model": name, "params": params, "weight bytes": wbytes,
                         "channels": n_ch, "weight bytes, all channels": wbytes * n_ch,
                         "peak activation bytes (est.)": act, "MACs / inference": macs,
                         "host latency [us] (measured, batch 1)": None if host is None else round(host * 1e6, 1),
                         "ESP32 latency [ms] (ESTIMATE)": f"{lat[0] * 1e3:.3f}-{lat[1] * 1e3:.3f}",
                         "ESP32 energy [uJ] (ESTIMATE)":
                             f"{lat[0] * V * I_MA[0] * 1e3:.2f}-{lat[1] * V * I_MA[1] * 1e3:.2f}"})
    df = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/resources.csv", index=False)
    note = ("\n\nESP32 numbers are ESTIMATES, not measurements: 240 MHz; int8 2-6 cycles/MAC and float32 "
            "4-10 cycles/MAC (assumed scalar C loops, no SIMD); +20 cycles per requantized output; "
            "30-68 mA at 3.3 V (ESP32 datasheet, modem-sleep, 240 MHz, radio off). Peak activation memory "
            "counts input + hidden buffers only. Host latency: this PC, NumPy integer reference vs PyTorch, "
            "batch 1, median of 2000 (Python overhead dominates; reference only).\n")
    Path("results/tables/resources.md").write_text(df.to_markdown(index=False) + note, encoding="utf-8")
    Path("results/resources.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
