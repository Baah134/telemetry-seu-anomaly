"""Port of telemanom's error processing (Hundman et al. 2018; khundman/telemanom,
telemanom/errors.py, master) with its config.yaml defaults: batch_size 70, window_size 30,
smoothing_perc 0.05 (EWM span 105), error_buffer 100, p 0.13, sd_lim 12, z from 2.5 to
11.5 in steps of 0.5, at most 5 sequences per window, flagged points < 50% of window.

This is telemanom's real detection pipeline: a nonparametric dynamic threshold recomputed
on sliding windows (30 x 70 = 2,100 points) of TEST errors, inverse-error thresholds,
minimum-error checks (0.05 absolute, 5% of the 5th-95th percentile range), a 100-point
error buffer and anomaly pruning. It uses no labels. It adapts to test errors, causally
at batch granularity (each batch of 70 is judged with the preceding 2,100 errors).

Differences from the original, all stated in docs/DECISIONS.md:
- l_s (sequence length) = our window W = 64 instead of 250 (used for the start-up
  smoothing fix and the number of initial points ignored).
- The C-2 special case is kept.
- np.delete indices are cast to int (newer NumPy rejects float indices).
- Divisions by a zero peak in pruning are guarded (treated as no separation).
Inputs are in the channel's ORIGINAL units (the absolute 0.05 checks depend on them).
"""
import numpy as np
import pandas as pd

CFG = dict(batch_size=70, window_size=30, smoothing_perc=0.05, error_buffer=100, p=0.13)
SD_LIM = 12.0


def _dilate(idx, n, b):
    """Indices within distance < b of any index in idx, clipped to [0, n). Equivalent to
    telemanom's concatenate(i, i+arange(1,b), i-arange(1,b)) + unique + clip, in O(n)."""
    m = np.zeros(n + 1, np.int64)
    if idx.size == 0:
        return idx
    lo = np.clip(idx - (b - 1), 0, n)
    hi = np.clip(idx + b, 0, n)
    np.add.at(m, lo, 1)
    np.add.at(m, hi, -1)
    return np.nonzero(np.cumsum(m[:n]) > 0)[0]


def _groups(idx):
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []
    cuts = np.nonzero(np.diff(idx) != 1)[0] + 1
    return [g for g in np.split(idx, cuts)]


class _Window:
    def __init__(self, e_s_all, y_test, start, end, num, n_test, l_s, cfg):
        self.cfg, self.num = cfg, num
        self.e_s = e_s_all[start:end]
        self.mean_e_s = float(np.mean(self.e_s))
        self.sd_e_s = float(np.std(self.e_s))
        self.e_s_inv = self.mean_e_s + (self.mean_e_s - self.e_s)
        self.epsilon = self.mean_e_s + SD_LIM * self.sd_e_s
        self.epsilon_inv = self.epsilon
        yt = y_test[start:end]
        self.sd_values = float(np.std(yt))
        hi, lo = np.percentile(yt, [95, 5])
        self.inter_range = float(hi - lo)
        self.num_to_ignore = l_s * 2
        if n_test < 2500:
            self.num_to_ignore = l_s
        if n_test < 1800:
            self.num_to_ignore = 0
        self.i_anom = np.array([], dtype=np.int64)
        self.i_anom_inv = np.array([], dtype=np.int64)
        self.E_seq, self.E_seq_inv = [], []
        self.non_anom_max = self.non_anom_max_inv = -1e6

    def find_epsilon(self, inverse=False):
        e_s = self.e_s_inv if inverse else self.e_s
        max_score = -1e7
        for z in np.arange(2.5, SD_LIM, 0.5):
            eps = self.mean_e_s + self.sd_e_s * z
            pruned = e_s[e_s < eps]
            i_anom = np.nonzero(e_s >= eps)[0]
            if i_anom.size == 0:
                continue
            i_anom = _dilate(i_anom, len(e_s), self.cfg["error_buffer"])     # offsets 1..buffer-1
            E_seq = [g for g in _groups(i_anom) if g[0] != g[-1]]
            with np.errstate(all="ignore"):
                mpd = (self.mean_e_s - np.mean(pruned)) / self.mean_e_s
                spd = (self.sd_e_s - np.std(pruned)) / self.sd_e_s
                score = (mpd + spd) / (len(E_seq) ** 2 + len(i_anom))
            if score >= max_score and len(E_seq) <= 5 and len(i_anom) < len(e_s) * 0.5:
                max_score = score
                if inverse:
                    self.epsilon_inv = self.mean_e_s + z * self.sd_e_s
                else:
                    self.epsilon = self.mean_e_s + z * self.sd_e_s

    def compare_to_epsilon(self, global_i_anom, inverse=False):
        e_s = self.e_s_inv if inverse else self.e_s
        eps = self.epsilon_inv if inverse else self.epsilon
        if not (self.sd_e_s > 0.05 * self.sd_values or self.e_s.max() > 0.05 * self.inter_range) \
                or not self.e_s.max() > 0.05:
            return
        i_anom = np.nonzero((e_s >= eps) & (e_s > 0.05 * self.inter_range))[0]
        if i_anom.size == 0:
            return
        i_anom = _dilate(i_anom, len(e_s), self.cfg["error_buffer"] + 1)     # offsets 1..buffer
        if self.num == 0:
            i_anom = i_anom[i_anom >= self.num_to_ignore]
        else:
            i_anom = i_anom[i_anom >= len(e_s) - self.cfg["batch_size"]]
        i_anom = np.unique(i_anom)
        pos = self.num * self.cfg["batch_size"]
        win_idx = np.arange(len(e_s)) + pos
        win_idx = np.setdiff1d(win_idx, np.append(global_i_anom, i_anom + pos))
        cand = np.unique(win_idx - pos)
        non_anom_max = float(np.max(e_s[cand])) if cand.size else -1e6
        E_seq = [g for g in _groups(i_anom) if g[0] != g[-1]]
        if inverse:
            self.i_anom_inv, self.E_seq_inv, self.non_anom_max_inv = i_anom, E_seq, non_anom_max
        else:
            self.i_anom, self.E_seq, self.non_anom_max = i_anom, E_seq, non_anom_max

    def prune(self, inverse=False):
        E_seq = self.E_seq_inv if inverse else self.E_seq
        e_s = self.e_s_inv if inverse else self.e_s
        nam = self.non_anom_max_inv if inverse else self.non_anom_max
        if len(E_seq) == 0:
            return
        mx = np.array([e_s[g[0]:g[-1] + 1].max() for g in E_seq])
        srt = np.append(np.sort(mx)[::-1], [nam])
        rm = np.array([], dtype=np.int64)
        for i in range(len(srt) - 1):
            sep = (srt[i] - srt[i + 1]) / srt[i] if srt[i] != 0 else 0.0
            if sep < self.cfg["p"]:
                rm = np.append(rm, np.nonzero(mx == srt[i])[0])
            else:
                rm = np.array([], dtype=np.int64)
        keep = [g for k, g in enumerate(E_seq) if k not in set(rm.astype(int).tolist())]
        if not keep:
            if inverse:
                self.i_anom_inv = np.array([], dtype=np.int64)
            else:
                self.i_anom = np.array([], dtype=np.int64)
            return
        idx_keep = np.concatenate([np.arange(g[0], g[-1] + 1) for g in keep])
        if inverse:
            self.i_anom_inv = self.i_anom_inv[np.isin(self.i_anom_inv, idx_keep)]
        else:
            self.i_anom = self.i_anom[np.isin(self.i_anom, idx_keep)]


def smooth(e, cfg=CFG, l_s=64, chan_id=None):
    span = int(cfg["batch_size"] * cfg["window_size"] * cfg["smoothing_perc"])
    e_s = pd.DataFrame(np.asarray(e, float)).ewm(span=span).mean().to_numpy().ravel().copy()
    if chan_id != "C-2" and len(e_s) > 0:
        e_s[:l_s] = np.mean(e_s[:l_s * 2])
    return e_s


def detect(e, y_test, l_s=64, chan_id=None, cfg=CFG):
    """e: raw |error| for each forecast step; y_test: actual values at the same steps
    (original units). Returns list of (start, end) sequences in the SAME index space as e."""
    e = np.asarray(e, float)
    y_test = np.asarray(y_test, float)
    n = len(e)
    if not np.all(np.isfinite(e)):
        e = np.where(np.isfinite(e), e, np.nan)
    e_s = smooth(e, cfg, l_s, chan_id)
    ws = cfg["window_size"]
    bs = cfg["batch_size"]
    n_windows = int((n - bs * ws) / bs)
    while n_windows < 0:
        ws -= 1
        n_windows = int((n - bs * ws) / bs)
        if ws == 1 and n_windows < 0:
            raise ValueError("batch_size larger than test series")
    if not np.all(np.isfinite(e_s)):
        # telemanom never sees NaN; on an MCU NaN comparisons are false -> no alarms
        e_s = np.where(np.isfinite(e_s), e_s, 0.0)
    g_anom = np.array([], dtype=np.int64)
    for i in range(n_windows + 1):
        prior = i * bs
        idx = ws * bs + i * bs if i < n_windows else n
        w = _Window(e_s, y_test, prior, idx, i, n, l_s, cfg)
        w.find_epsilon()
        w.find_epsilon(inverse=True)
        w.compare_to_epsilon(g_anom)
        w.compare_to_epsilon(g_anom, inverse=True)
        if len(w.i_anom) == 0 and len(w.i_anom_inv) == 0:
            continue
        w.prune()
        w.prune(inverse=True)
        if len(w.i_anom) == 0 and len(w.i_anom_inv) == 0:
            continue
        ia = np.unique(np.append(w.i_anom, w.i_anom_inv)).astype(np.int64)
        g_anom = np.append(g_anom, ia + prior)
    if g_anom.size == 0:
        return []
    g_anom = np.unique(g_anom)
    return [(int(g[0]), int(g[-1])) for g in _groups(g_anom) if g[0] != g[-1]]
