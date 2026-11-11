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
