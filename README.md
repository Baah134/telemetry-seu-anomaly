# Tiny onboard anomaly detection for spacecraft telemetry: honest metrics and simulated SEUs

Simulation study. Can a ~4 KB INT8 forecaster detect anomalies in NASA spacecraft
telemetry as well as a 350 KB LSTM, when it is evaluated honestly? And what happens to
it, and to its float32 twin, when bits flip in memory (simulated single-event upsets)?

- Data: NASA SMAP and MSL telemetry with expert labels (Hundman et al., KDD 2018), the
  official telemanom release verified by SHA-256 (docs/DATA.md).
- Models: tiny MLP forecaster (3.4k–4.4k parameters) in float32 and as an integer-only
  INT8 model (TFLite-Micro-style arithmetic, NumPy reference + C for ESP32); LSTM 2x80
  reference (telemanom size); persistence, rolling z-score and random-score baselines.
- Evaluation: event-wise precision/recall/F1 (headline), point-wise F1 without
  adjustment, false alarms per 1,000 steps, and point-adjusted F1 shown only next to
  random scores. Detection uses telemanom's own label-free pipeline (ported), plus two
  fixed validation thresholds for comparison.
- Faults (simulated, not radiation testing): persistent weight bit flips, bit-position
  and layer sensitivity, transient accumulator flips, and four lightweight protections.

**Results: [RESULTS.md](RESULTS.md).** Decisions: [docs/DECISIONS.md](docs/DECISIONS.md).
Study guide: [docs/EXPLAINER.md](docs/EXPLAINER.md).

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-lock.txt
make data      # downloads + SHA-256-checks the official zip (~86 MB)
make test      # pytest
make figures   # tables and figures from committed results (minutes)
make all       # everything from scratch (several hours on 12 threads)
```
Without `make`: `python reproduce.py data test figures`. The host C check
(`make export`) needs gcc on PATH. ESP32 steps: [export/HARDWARE_TEST.md](export/HARDWARE_TEST.md).

## Layout

`src/data` loading and windowing · `src/detect` metrics, thresholds, telemanom port ·
`src/models` tiny MLP, LSTM · `src/quant` INT8 quantization and integer inference ·
`src/faults` bit flips and protections · `scripts/` experiments and summaries ·
`results/` all result files · `figures/` · `export/` ESP32 sketch · `tests/`.

## Limitations

Simulation only: no radiation or beam testing, no on-device fault injection. On-device:
INT8 inference measured on one ESP32 DevKit (bit-exact, 203 µs); energy is estimated. One benchmark (SMAP/MSL), which has known
flaws (dense, often trivially visible anomalies). The fault study uses 10 channels with
13 labeled anomalies, so its F1 values move in steps of about 0.08.
