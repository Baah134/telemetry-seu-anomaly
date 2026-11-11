# EXPLAINER — study guide for Prince (Project 3)

Read this before you write anything about the project. Sections 7–8 are filled in from
the results files once the runs finish.

## 1. SMAP, MSL and the telemetry

- **SMAP** (Soil Moisture Active Passive) is a NASA Earth-observation satellite (launched
  2015). **MSL** is the Mars Science Laboratory, the Curiosity rover.
- Hundman et al. (KDD 2018) released 82 telemetry channels (55 SMAP, 27 MSL). Each is one
  sensor or subsystem value over time, already scaled (mostly to [-1, 1]) and anonymised
  (channel IDs like P-1, T-1), plus one-hot "command" columns saying which commands were
  sent to the spacecraft at each step (24 for SMAP, 54 for MSL).
- Anomalies were labeled by spacecraft experts from incident-surprise-anomaly (ISA) reports:
  start/end indices in the test segment, each tagged *point* (a value far outside normal)
  or *contextual* (a value that is normal in range but wrong for the situation).
- Facts you can quote: 105 labeled sequences (69 SMAP, 36 MSL); 12.5% of test points are
  labeled anomalous; most channels have exactly one anomaly. See docs/DATA.md.

## 2. Forecasting-based anomaly detection, step by step

1. **Forecast.** A model reads the last W = 64 values (plus the latest command vector) and
   predicts the next value. It is trained only on the training segment, where we assume
   behaviour is mostly nominal.
2. **Error.** At each test step, e[t] = |y[t] − ŷ[t]|. A big error means the spacecraft did
   something the model did not expect.
3. **Smooth.** Raw errors are spiky, so we use an exponentially weighted moving average,
   s[t] = 0.05·e[t] + 0.95·s[t−1]. This is what telemanom does too.
4. **Threshold.** Flag s[t] > ε. ε must be chosen *without test labels*:
   - **percentile:** ε = the 99.5th percentile of s on the validation split (the last 20%
     of the training series, kept in time order).
   - **NDT (Hundman's nonparametric dynamic threshold):** try ε = μ + zσ for z = 2…10 and
     keep the ε that removes the most mean and spread from the errors while flagging the
     fewest points and runs. Telemanom recomputes this on sliding windows of test errors;
     we fit it once on validation errors (simpler, and test errors cannot influence it).
5. **Alarm.** Contiguous flagged points form one alarm event.

The baselines use the same steps: **persistence** predicts ŷ[t] = y[t−1] (no learning at
all), the **rolling z-score** scores how unusual y[t] is relative to the previous 64
points, and **random scores** are i.i.d. uniform noise.

## 3. Metrics, and why point adjustment inflates scores

- **Event-wise (headline).** A labeled anomaly is detected if any alarm overlaps it. Every
  alarm event that overlaps no anomaly is one false positive. This matches how an operator
  experiences alarms.
- **Point-wise, no adjustment.** Each timestep is a sample: precision = flagged anomalous
  points / flagged points; recall = flagged anomalous points / anomalous points.
- **Point-adjusted (PA).** If *any* point inside a labeled anomaly is flagged, *every*
  point of that anomaly counts as flagged, then point-wise F1 is computed.

**Tiny worked example.** 1,000 timesteps, one anomaly of 100 points (steps 400–499). A
detector flags one point at step 450 and nothing else.
- Point-wise: TP = 1, FP = 0, FN = 99 → precision 1.0, recall 0.01, F1 ≈ 0.02.
- Point-adjusted: the single hit turns into 100 hits → TP = 100, FN = 0 → F1 = 1.0.
So one lucky alarm becomes a perfect score. With long anomalies and 12.5% anomalous
points, even **random** alarms hit most anomalies somewhere, and PA rewards them
(Kim et al., AAAI 2022, prove this effect). That is why PA-F1 is only ever shown next to
the random-score baseline here.

**Benchmark critiques.** Wu & Keogh (2021, "Current time series anomaly detection
benchmarks are flawed and are creating the illusion of progress") argue that many
benchmark anomalies, SMAP/MSL included, are trivially detectable, some labels are
questionable, and anomaly density is unrealistic. We test this directly: the
"benchmark sanity" table shows how many anomalies persistence and z-score already catch.

**The parallel with your SER audit.** There, speaker-leaked splits made published
accuracies look 6.9–63.1 points better than speaker-independent evaluation. Here, point
adjustment and anomaly-dense benchmarks play the same role: the evaluation protocol, not
the model, produces much of the headline number.

## 4. SEUs: what they are and why they matter

- A **single-event upset** is a bit flip caused when an energetic charged particle (a
  galactic cosmic ray, a solar proton, or a particle trapped in the Van Allen belts, e.g.
  the South Atlantic Anomaly) passes through a transistor and deposits enough charge to
  flip a stored 0/1.
- **Memory vs logic.** SRAM and registers hold state, so a flip *persists* until the value
  is rewritten (our "persistent weight upsets"). A strike in combinational logic or in a
  value that is about to be overwritten causes a *transient* error that affects one
  computation (our "transient activation upsets").
- Small satellites often fly commercial microcontrollers without radiation hardening or
  ECC memory, which is why lightweight software protections are interesting.
- **This project simulates bit flips in software.** It is not radiation testing.

## 5. INT8 quantization arithmetic

- A real value r is stored as an integer q with a scale s: r ≈ s·q (symmetric, zero
  point 0). For weights, s = max|W_row| / 127 per output neuron; for activations, one s per
  layer from the largest value seen on training data.
- A layer computes acc = Σ W_q·x_q + b_q in **int32**, which equals
  (Σ W·x + b) / (s_x·s_w). To get the next layer's int8 input it is rescaled by
  M = s_x·s_w / s_y and rounded: y_q = round(acc·M), saturated to [−127, 127] (ReLU →
  [0, 127]).
- MCUs avoid floating point here: M is stored as a 31-bit integer M0 and a shift, so
  acc·M = (acc·M0) >> 31 >> shift, done with integer multiplies (as in TFLite Micro).
- **Why int8 and float32 fail differently.** Any bit flip leaves an int8 weight inside
  [−128, 127], so the corrupted weight is at most about as large as the row's largest
  weight (the sign bit changes it by exactly 128 steps): bounded.
  Flipping a high **exponent** bit of a float32 can multiply the value by 2^128 or turn it
  into inf/NaN: unbounded. Int32 biases and accumulators sit in between (bit 30 is ±2^30
  steps). The fault study measures how much this matters for detection.

## 6. The protections

| Protection | How it works | Cost | When it fails |
|---|---|---|---|
| Range restriction | clamp each layer's output to the range seen on training data | 2 bounds per layer, 1 compare per activation | errors that stay inside the normal range are not caught |
| Checksum + scrub | CRC32 per weight array, recomputed every K inferences; on mismatch reload from a golden copy | 4 bytes per array + reading all weight bytes every K inferences | wrong outputs until the next check (latency ≈ K/2 inferences on average); golden copy must itself be protected |
| Selective TMR | store one layer 3 times, use the bitwise majority | 2× that layer's bytes, vote per read | two copies hit in the same bit; the other layers stay unprotected |
| Output sanity check | if a forecast is far outside the training range, use the last value instead | 2 bounds, 2 compares | corrupted forecasts that still look plausible |
