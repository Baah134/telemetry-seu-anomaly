import numpy as np
import pandas as pd
import pytest


def make_synth(root, n_chan=3, seed=0):
    """Tiny synthetic telemanom-format dataset: sinusoid + commands, injected spikes."""
    rng = np.random.default_rng(seed)
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    rows = []
    for i in range(n_chan):
        cid = f"X-{i}"
        n_cmd = 4
        def series(n, anomalies=()):
            t = np.arange(n)
            v = 0.5 * np.sin(2 * np.pi * t / 100 + i) + 0.02 * rng.normal(size=n)
            for s, e in anomalies:
                v[s:e + 1] += 1.5
            c = (rng.random((n, n_cmd)) > 0.97).astype(float)
            return np.column_stack([np.clip(v, -1, 1), c])
        np.save(root / "train" / f"{cid}.npy", series(2000))
        an = [(600, 640), (1200, 1210)]
        np.save(root / "test" / f"{cid}.npy", series(1500, an))
        rows.append({"chan_id": cid, "spacecraft": "SMAP" if i % 2 == 0 else "MSL",
                     "anomaly_sequences": str([list(a) for a in an]),
                     "class": "[point, contextual]", "num_values": 1500})
    pd.DataFrame(rows).to_csv(root / "labeled_anomalies.csv", index=False)
    return root


@pytest.fixture
def synth(tmp_path):
    return make_synth(tmp_path / "raw")
