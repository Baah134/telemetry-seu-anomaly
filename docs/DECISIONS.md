# DECISIONS — Project 3 (telemetry anomaly detection + simulated SEU fault injection)

## Session 1 — 2026-11-11 (setup, M0/M1 code, fault harness)

Status: code for M0–M4 written and smoke-tested on a synthetic dataset; real data pending.

1. **Data source.** The official telemanom S3 link (s3-us-west-2.amazonaws.com/telemanom/data.zip)
   returns 403. The telemanom README now points to Kaggle
   (patrickfleith/nasa-anomaly-detection-dataset-smap-msl), which needs Prince's Kaggle
   login, so Prince downloads it. `labeled_anomalies.csv` comes from the official
   khundman/telemanom repo: 82 channels (55 SMAP, 27 MSL), matching the paper.
2. **Python 3.14** venv (spec says 3.11; 3.11 is not installed on this machine). Code avoids
   3.12+ syntax so it runs on 3.11.
3. **Tiny model = MLP, not 1D-CNN.** [64 values + n_cmd commands] -> 32 -> 16 -> 1 (ReLU).
   3.4k params (SMAP, 24 commands) to 4.4k (MSL, 54 commands): under the 5k / 5 KB target.
   Three int8 GEMVs keep the integer reference and the fault study transparent.
4. **Window W = 64** for every method (telemanom used 250). Same window for the LSTM, so
   the tiny-vs-LSTM comparison isolates model capacity. The first 64 test points get no
   forecast and are never flagged, for every method.
5. **LSTM reference: 2 x 80 (as telemanom)**, dropout 0.3, early stopping, max 25 epochs.
6. **Validation split = last 20% of each training series** (chronological). Normalization
   statistics come from the first 80% only; thresholds from the validation 20% only.
7. **Dynamic threshold.** Hundman's NDT objective implemented exactly (z grid 2..10 step
   0.5), but fitted once on validation errors and applied as a fixed threshold, instead
   of telemanom's sliding windows over test errors. Reason: a clean no-leakage story
   (test errors never influence the threshold). Also no telemanom pruning step. Both
   simplifications are stated in RESULTS.
8. **Percentile threshold = 99.5th percentile of validation smoothed errors.** Chosen
   a priori and used as the fault-study threshold before seeing any test results.
9. **Event-wise metric = telemanom definition**; FP events = contiguous predicted runs
   with no overlap with any label. Aggregation is micro (sum counts over channels).
10. **INT8 scheme:** symmetric, per-output-channel weights, per-tensor activations
    calibrated on training windows, int32 bias, TFLite-style fixed-point requantization
    (Q31 multiplier + rounding shift). Integer GEMVs are computed exactly with float64
    BLAS (|sum| < 2^53).
11. **Fault primary axis = number of flipped bits in weight memory** (1..100), not BER:
    a single model is only ~3.5–4.6 KB (28k–37k bits), so a BER of 1e-5 means < 1
    expected flip and BER sweeps mostly measure the probability of zero flips. k flips
    are uniform over all stored bits (int8 weights and int32 biases; float32 for float).
12. **Fault subset:** 5 SMAP + 5 MSL channels drawn at random (seed 0) from channels
    with labeled anomalies, fixed before any fault result. Model training seed 0.
13. **NaN handling:** a float32 exponent flip can produce inf/NaN. NaN errors never exceed
    a threshold (IEEE comparison), so a NaN forecast silently suppresses alarms, which is
    what an MCU comparison would also do. Counted and reported (`nonfinite`).
14. **TMR layer chosen from a label-free sensitivity** (median change of validation
    forecasts under 1 flip per layer/bit), never from test F1.
15. **Protection timing:** in the protection study, k flips occur at a random time t0 in
    the test stream and persist (or until the next scrub check). In the sweep they are
    present for the whole stream (worst case).
16. **Data obtained from the Internet Archive snapshot of the official S3 zip**, verified
    byte-identical by SHA-256 against the checksum published by TimeSeAD
    (b4d66deb...d8a7). Prince could not access Kaggle. No re-processed mirror was used.
17. **Constant training series (M-6)**: std = 0, so normalization falls back to sd = 1
    (loader `or 1.0`). Its test series jumps to 258, a trivially visible anomaly.
18. **P-2 duplicate merged.** labeled_anomalies.csv lists P-2 twice (overlapping labels);
    merged into one channel with the label union: 81 channels, 104 sequences (68 SMAP).
19. **Telemanom's own detection pipeline added as the headline detector** (src/detect/
    telemanom.py, a port of telemanom/errors.py with its config defaults). Reason: the
    simplified fixed thresholds gave every method ~3 false-alarm events per 1,000 steps
    (event precision 0.05-0.15), which made the tiny model look worse than persistence;
    telemanom's windowed NDT + pruning + minimum-error checks are part of the reference
    method and use no labels. Both views are reported: the ranking flips between them,
    which is itself a finding about evaluation protocol. Differences from the original:
    l_s = 64 (our window) instead of 250; exact O(n) re-implementation of the error-buffer
    expansion (tested equal to the original expansion); int-cast delete indices; guarded
    zero-peak division. A line-by-line parity test against the original telemanom code
    was NOT run (the sandbox blocked executing downloaded code); behavioural tests are in
    tests/test_telemanom.py. Applies to forecasters only (persistence, tiny, LSTM); the
    z-score and random baselines use the fixed thresholds.
20. **Fault study switched from the percentile threshold (decision 8) to the telemanom
    pipeline**, after seeing the clean results: under the fixed threshold the clean
    detector's event precision is ~0.05, so fault effects on F1 would be uninterpretable.
    This was decided from clean results only, before any fault result existed.
21. **Fault conditions trimmed for compute** (telemanom detection ~0.5 s per channel):
    protections at k = 10 and 100 flips (not 1); scrub@K sweep for int8 at k = 10 only;
    bit-position study reports label-free forecast deviation only (no detection metric).
22. **TMR tested on two layers.** Per flipped bit the output layer (L2) is ~20-70x more
    sensitive than L0/L1 (validation forecast deviation), but L0 holds ~90% of the weight
    bits, so it receives most random upsets. Both TMR(L2) and TMR(L0) are evaluated.
23. **ESP32 export verified on the host:** the sketch's C arithmetic compiled with gcc
    is bit-exact with the NumPy integer reference on 1,000/1,000 windows (channel D-2,
    results/c_host_check.json). This is not an ESP32 measurement.
24. **LSTM kept at full telemanom size (2 x 80)**, max 25 epochs, early stopping
    (patience 4) on the validation split; ~21 s per epoch per channel on this machine.
25. **Forecast-damage experiment added** (results/faults/damage.json): same flips as the
    sweep (identical RNG seeds), forecast MSE outside labeled anomalies relative to the
    clean model. Added because INT8 event F1 rose with more flips; this checks whether F1
    tracks model health (it does not).
26. **Protection study rerun after two bugs were found in the first run (not reported):**
    (a) every scrub@K variant recorded the K = 100 corrupted-inference count;
    (b) after a NaN forecast the detector's EWMA stayed poisoned for the rest of the
    stream even after scrubbing repaired the weights. Now scrubbing also resets the
    detector state at the repair step (what an implementation would do); without
    scrubbing the NaN still silences the detector permanently.
27. **No-fault baselines per protection.** Clamp and the sanity check act on every
    inference, not only faulty ones, so each protection's event F1 without faults is
    reported (results/faults/clean.json, `*_no_fault_with`). Finding: the sanity check
    lowers fault-free F1 (INT8 0.615 -> 0.560) because it also replaces real
    out-of-range forecasts during anomalies; the float32 activation clamp RAISES
    fault-free F1 (0.545 -> 0.640), to about the INT8 level.
28. **Scrubbing modelled as verify-then-commit.** In the second protection run, float32
    with CRC scrubbing (K = 100) was WORSE than no protection at 100 flips (F1 0.368 vs
    0.428, 19% catastrophic): the <= 100 corrupted forecasts before the check are up to
    ~1e30, so their error burst raised a false alarm and kept the smoothed error inflated
    for thousands of steps, hiding real anomalies. That is a real failure mode of naive
    scrubbing. The design used now: forecasts are released to the detector only after
    the next checksum passes; on a mismatch, all forecasts since the last good check are
    discarded and the weights reloaded. Cost: alarm latency of up to K inferences. Both
    the naive result (this entry) and the verify-then-commit result are reported.
29. **T-10 unused.** The official zip contains 82 channel files; T-10 has no row in
    labeled_anomalies.csv (no ground truth), so the evaluation uses the 81 labeled
    channels. Found during the fresh-clone check.
30. **Fresh-clone check (2026-11-11):** found that `.gitignore`'s `data/` rule also
    excluded `src/data/` (the loader); fixed to `/data/`. After the fix a fresh clone
    passes all tests, `get_data` verifies the zip, and the summary scripts regenerate
    every table and RESULTS.md identically (line endings aside).

## Session 1 end-of-day — 2026-11-11
Done: M0–M5 in simulation (data, metrics, baselines, tiny + LSTM, INT8 + integer
reference + C export, fault study with protections, docs). Found: see RESULTS.md.
Open: ESP32 measurement (Prince); telemanom side-by-side parity run (needs permission
to execute the original code); optional shared-model variant and QAT not done.
