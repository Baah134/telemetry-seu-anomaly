"""Run Makefile targets without make (Windows). Usage: python reproduce.py [target ...]"""
import re
import subprocess
import sys
from pathlib import Path

mk = Path(__file__).with_name("Makefile").read_text(encoding="utf-8")
rules, cur = {}, None
for line in mk.splitlines():
    m = re.match(r"^([A-Za-z][\w-]*):(.*)$", line)
    if m and not line.startswith("\t"):
        cur = m.group(1)
        rules[cur] = {"deps": m.group(2).split(), "cmds": []}
    elif line.startswith("\t") and cur:
        rules[cur]["cmds"].append(line.strip())
done = set()


def run(t):
    if t in done:
        return
    for d in rules[t]["deps"]:
        run(d)
    for c in rules[t]["cmds"]:
        if c.startswith("rm "):
            continue
        c = c.replace("$(RUN)", f'"{sys.executable}" -W ignore -m').replace("$(PY)", f'"{sys.executable}"')
        print(">", c, flush=True)
        subprocess.run(c, shell=True, check=True)
    done.add(t)


for t in sys.argv[1:] or ["all"]:
    run(t)
