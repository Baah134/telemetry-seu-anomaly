# Hardware test on ESP32 (Prince runs this; nothing here has been measured yet)

Goal: check that the INT8 model runs **bit-exactly** on the ESP32 like the NumPy integer
reference, and measure real latency (and energy, if you have a power meter). Until you do
this, every ESP32 number in the repo is an **estimate** (results/tables/resources.md).

## 1. Generate the model header

```bash
python -m scripts.export_c            # writes export/esp32_infer/model_data.h
```
It also writes `export/expected_outputs.json` (the reference outputs for the stored windows).

## 2. Flash

1. Arduino IDE 2.x → Boards Manager → install **esp32 by Espressif Systems**.
2. Open `export/esp32_infer/esp32_infer.ino`. Select your board (e.g. *ESP32 Dev Module*),
   CPU frequency **240 MHz**, and the right COM port.
3. Upload, then open the Serial Monitor at **115200** baud. Press EN/RST if nothing prints.

## 3. What you should see

```
idx,expected,got,match,us_per_inference
0,<n>,<n>,1,<time>
...
bit-exact matches: 8 / 8
```
- `match` must be 1 for all rows. If any row is 0, report it: it means the C arithmetic
  differs from the NumPy reference (that is a finding, not something to hide).
- `us_per_inference` is the mean of 1,000 back-to-back inferences on one stored window.

## 4. Energy (optional)

With a USB power meter or a shunt + oscilloscope: measure the board current while the
loop runs vs. idle (`delay`). Energy per inference ≈ (I_run − I_idle) × V × t_inference.
State the meter, the supply voltage and whether the dev board's regulator/LED are included.

## 5. Record the result

Send me the Serial output and how you measured, and I'll add
`results/hardware_measured.json` with the date, board, CPU frequency and method. Only then
will the docs say "measured by Prince on <board>".

## What this does NOT test

Radiation effects. Fault injection in this repo is simulated bit flips in software.
Real SEU testing needs beam time (proton or heavy-ion) or at least on-device fault
injection (flipping bits in the ESP32's RAM while it runs).
