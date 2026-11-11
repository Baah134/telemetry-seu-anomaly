"""Compile the ESP32 sketch's integer inference with a host C compiler and check it is
bit-exact against the NumPy integer reference on N windows (default 1,000).
This checks the C arithmetic, not the ESP32 itself (see export/HARDWARE_TEST.md).
Usage: python -m scripts.check_c_host [--n 1000] [--cc gcc]
Writes results/c_host_check.json.
"""
import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HOST_MAIN = """#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "core.c"
int main(void) {
  int ok = 0;
  for (int k = 0; k < N_TEST; k++) {
    int8_t y = infer(TEST_X + k * IN_DIM);
    ok += (y == TEST_Y_EXPECTED[k]);
  }
  printf("%d %d\\n", ok, N_TEST);
  return 0;
}
"""


def main():
    from scripts.export_c import build
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--cc", default=shutil.which("gcc") or "gcc")
    a = ap.parse_args()
    chan, qm, idx, yq, header = build(None, a.n)
    ino = Path("export/esp32_infer/esp32_infer.ino").read_text(encoding="utf-8")
    core = ino[ino.index('#include "model_data.h"'):ino.index("void setup()")]
    core = core.replace('#include "model_data.h"', '#include "model_data.h"\n#include <stdint.h>\n#include <stdbool.h>')
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "model_data.h").write_text(header, encoding="utf-8")
        (d / "core.c").write_text(core, encoding="utf-8")
        (d / "host.c").write_text(HOST_MAIN, encoding="utf-8")
        subprocess.run([a.cc, "-O2", "-std=c11", "-Wall", "-o", str(d / "host.exe"), str(d / "host.c")], check=True)
        out = subprocess.run([str(d / "host.exe")], capture_output=True, text=True, check=True).stdout
    ok, n = map(int, re.findall(r"\d+", out)[:2])
    rec = {"chan": chan, "windows": n, "bit_exact": ok, "distinct_outputs": len(set(yq.tolist())),
           "compiler": a.cc, "note": "host C compiler, not the ESP32"}
    Path("results/c_host_check.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(rec)


if __name__ == "__main__":
    main()
