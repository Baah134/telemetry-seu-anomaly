// INT8 tiny-MLP forecaster for ESP32 (Arduino core). Runs the same integer arithmetic as
// src/quant/int8.py on stored test windows, checks the outputs against the NumPy
// reference (TEST_Y_EXPECTED) and prints the elapsed microseconds per inference.
// Generate model_data.h with:  python -m scripts.export_c
#include "model_data.h"

static int32_t srdhm(int32_t a, int32_t b) {           // SaturatingRoundingDoublingHighMul
  if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
  int64_t ab = (int64_t)a * (int64_t)b;
  int64_t nudge = ab >= 0 ? (1LL << 30) : (1 - (1LL << 30));
  int64_t num = ab + nudge;
  return (int32_t)(num >= 0 ? (num >> 31) : -((-num) >> 31));
}

static int32_t rshift_round(int32_t x, int sh) {
  if (sh <= 0) return x;
  int32_t mask = (1 << sh) - 1;
  int32_t rem = x & mask;
  int32_t thr = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> sh) + (rem > thr ? 1 : 0);
}

static int32_t requant(int32_t acc, int32_t m0, int8_t sh) {
  if (sh < 0) {                                          // multiplier >= 1
    int64_t v = (int64_t)acc << (-sh);
    acc = v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : (int32_t)v);
    sh = 0;
  }
  return rshift_round(srdhm(acc, m0), sh);
}

static void dense(const int8_t* x, int in, const int8_t* W, const int32_t* B, const int32_t* M,
                  const int8_t* S, int out, int8_t* y, bool relu) {
  for (int j = 0; j < out; j++) {
    int32_t acc = B[j];
    const int8_t* w = W + j * in;
    for (int i = 0; i < in; i++) acc += (int32_t)w[i] * (int32_t)x[i];
    int32_t v = requant(acc, M[j], S[j]);
    int32_t lo = relu ? 0 : -127;
    if (v < lo) v = lo;
    if (v > 127) v = 127;
    y[j] = (int8_t)v;
  }
}

static int8_t infer(const int8_t* x) {
  static int8_t h1[H1], h2[H2], o[1];
  dense(x, IN_DIM, L1_W, L1_B, L1_M, L1_S, H1, h1, true);
  dense(h1, H1, L2_W, L2_B, L2_M, L2_S, H2, h2, true);
  dense(h2, H2, L3_W, L3_B, L3_M, L3_S, 1, o, false);
  return o[0];
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("idx,expected,got,match,us_per_inference");
  int ok = 0;
  for (int k = 0; k < N_TEST; k++) {
    const int8_t* x = TEST_X + k * IN_DIM;
    const int REPS = 1000;
    volatile int8_t y = 0;
    uint32_t t0 = micros();
    for (int r = 0; r < REPS; r++) y = infer(x);
    uint32_t dt = micros() - t0;
    bool m = (y == TEST_Y_EXPECTED[k]);
    ok += m;
    Serial.printf("%d,%d,%d,%d,%.3f\n", k, TEST_Y_EXPECTED[k], (int)y, m, dt / (float)REPS);
  }
  Serial.printf("bit-exact matches: %d / %d\n", ok, N_TEST);
}

void loop() {}
