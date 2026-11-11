import numpy as np

from src.data.loader import W, load_all, windows


def test_loader_shapes_labels_split(synth):
    chans = load_all(raw=synth)
    assert len(chans) == 3
    for c in chans:
        assert c.train.shape[1] == c.test.shape[1] == 5
        for s, e in c.label_seqs:
            assert 0 <= s <= e < len(c.test)
        fit, val = c.split()
        assert len(fit) + len(val) == len(c.train)
        # chronological: validation is the END of the training series
        assert np.array_equal(val, c.train[len(fit):])
        X, y = windows(c.test, c)
        assert X.shape == (len(c.test) - W, W + c.n_cmd) and y.shape == (len(c.test) - W,)
        # window t ends at t-1 and the target is t
        v = (c.test[:, 0] - c.mu) / c.sd
        assert np.allclose(X[10, W - 1], v[W + 9]) and np.allclose(y[10], v[W + 10])


def test_normalization_uses_fit_part_only(synth):
    c = load_all(raw=synth)[0]
    fit, _ = c.split()
    assert np.isclose(c.mu, fit[:, 0].mean()) and np.isclose(c.sd, fit[:, 0].std())
