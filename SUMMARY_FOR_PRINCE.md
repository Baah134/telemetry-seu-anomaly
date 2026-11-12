# Summary for Prince — Project 3 (facts only; write your own words from these)

**What it is.** A simulation study of onboard anomaly detection for spacecraft telemetry
on NASA SMAP/MSL data (Hundman et al., KDD 2018). It asks two trust questions: is a tiny
INT8 detector actually good under honest evaluation, and what happens when bits flip in
its memory (simulated single-event upsets)?
Repo (private): https://github.com/Baah134/telemetry-seu-anomaly

**What you built (with Claude Code).**
- Data pipeline on the official telemanom release, verified by SHA-256; found and fixed
  a duplicated channel (P-2) in the official label file.
- Honest metrics: event-wise F1 (headline), point-wise F1 without adjustment, false
  alarms per 1,000 steps; point-adjusted F1 only next to random scores.
- A port of telemanom's detection pipeline (windowed dynamic threshold + pruning).
- Tiny MLP forecaster (3.4k–4.4k parameters), INT8 post-training quantization, an
  integer-only inference reference (TFLite-Micro-style fixed-point requantization), and
  a C version for ESP32 that is bit-exact with it on 1,000/1,000 windows (host gcc).
- LSTM 2x80 reference; persistence, rolling z-score and random baselines; 3 seeds.
- Bit-flip fault injection (persistent weight flips, transient accumulator flips,
  bit-position/layer sensitivity) with four protections; 100 Monte Carlo trials per
  condition; 24 tests.

**Key numbers (simulation, 81 channels, see RESULTS.md for all).**
- Tiny INT8 model: event F1 0.689 ± 0.008 vs LSTM 0.699 ± 0.019 (difference inside the
  seed spread) with 85–97x less weight memory (3.5–4.5 KB vs 345–383 KB per channel).
- Evaluation protocol flips the ranking: with a fixed threshold, persistence beats the
  tiny model (event F1 0.174 vs 0.109); with telemanom's pipeline the tiny model wins
  (0.689 vs 0.598). Random scores get point-adjusted F1 0.926, the best of any method.
- Float32 vs INT8 under 100 flipped weight bits: 98% of float32 trials produce NaN/inf
  forecasts, which silence the detector; INT8 cannot produce NaN and had 0% catastrophic
  trials. Transient upsets at 1 per 100 inferences: 47% catastrophic (float32) vs 0%
  (INT8).
- Detection F1 does not reveal model damage: after 100 flips, 44% of INT8 models have
  more than 2x their normal forecast error, yet event F1 does not drop (0.713 vs 0.615
  clean; only 13 labeled events in the fault subset, so steps of ~0.08).
- Protections (100 flips, 10-channel subset, clean F1: INT8 0.615, float32 0.545):
  - float32: a 24-byte activation range clamp gives F1 0.638 (vs 0.428 unprotected), but
    it also raises fault-free F1 to 0.640, so part of that is not fault protection.
    CRC32 scrubbing every 100 inferences (24 bytes) recovers 0.563 with 0% catastrophic
    trials, but only when outputs are released after the checksum passes; a naive
    version (outputs used immediately) did worse than no protection in an earlier run
    (logged in docs/DECISIONS.md #28, not in the final results files). TMR on the output layer did not help
    (0.424); TMR on the input layer did (0.538) but costs 26 KB, ~2x the model.
  - INT8: no protection changed event F1 beyond the trial spread (0.62–0.63), because
    faults did not break detection in the first place; the output sanity check costs
    about 0.05 F1 even without faults (it hides real out-of-range anomalies).

**One-sentence takeaway you can verify.** On SMAP/MSL, a 4 KB INT8 detector matched a
350 KB LSTM under the reference method's own pipeline, but the choice of metric and
threshold changed which model looked best, and the detector's score stayed high even
when bit flips had badly damaged the model.

**Link to your SER audit.** Same lesson: the evaluation protocol (speaker-leaked splits
there; point adjustment and thresholding here) can move the headline number more than
the model does.

**Words to avoid.** "radiation-tested", "radiation-hardened", "flight-qualified",
"runs on the ESP32 at X ms" (until you measure it), "INT8 is more robust in general".
Say "simulated bit flips", "estimated", "on this benchmark".

**What hardware validation would require.**
1. Flash `export/esp32_infer` (export/HARDWARE_TEST.md): confirm bit-exact outputs and
   measure latency/energy on your ESP32.
2. On-device fault injection: flip bits in the ESP32's RAM while the model runs.
3. Radiation: proton or heavy-ion beam time at a test facility, with upset logging.

**Things to be able to explain.** Why point adjustment inflates scores (worked example
in docs/EXPLAINER.md §3); what telemanom's dynamic threshold does; INT8 scale /
requantization; why a float32 exponent flip is catastrophic and an int8 flip is bounded;
why NaN silences a threshold detector; what each protection costs.
