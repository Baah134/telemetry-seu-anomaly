# DATA — NASA SMAP / MSL telemetry (Hundman et al., KDD 2018)

## Source and integrity

- `data.zip`: the official telemanom release. The original link
  (`https://s3-us-west-2.amazonaws.com/telemanom/data.zip`) returns HTTP 403 as of
  2026-11-11, and the telemanom README now points to a Kaggle copy that needs a login.
  The file was downloaded from the Internet Archive's snapshot of the original S3 URL
  (`web.archive.org/web/20221016205142/.../telemanom/data.zip`).
- **Integrity:** SHA-256 `b4d66deb492d9b0a353b51879152687ed9313897e8e19320d2dc853d738ed8a7`,
  identical to the checksum of the official zip published independently by the TimeSeAD
  library (`timesead/data/smap_dataset.py`, `ZIP_CHECKSUM`). So the file is byte-identical
  to the official release, not a re-processed mirror.
- `labeled_anomalies.csv`: from github.com/khundman/telemanom (master).
- The zip also contains telemanom's pretrained Keras models (`data/2018-05-19_15.00.10/`);
  they are not used.
- Not committed (gitignored). Reproduce with `make data`.

## Structure (verified on load by `src/data/loader.py`)

Per channel: `train/<id>.npy` and `test/<id>.npy`, shape (n, 1 + n_cmd). Column 0 is the
telemetry value (already scaled by NASA, mostly within [-1, 1]); the other columns are
one-hot command features: 24 for every SMAP channel, 54 for every MSL channel.
`labeled_anomalies.csv` gives inclusive [start, end] anomaly sequences in test indices.
The loader checks that train/test have the same width, that the test length equals
`num_values`, and that every label lies inside the test range. All channels pass.

**Duplicate row.** The CSV has 82 rows but 81 unique channels: P-2 (SMAP) appears twice,
with overlapping labels [5350, 6575] and [5300, 6420]. The loader merges duplicate rows
into one channel with the union of their labels ([5300, 6575]). Counting P-2 twice would
double-count its test points and its anomaly. After merging: 81 channels, 104 sequences.

## Summary (`python -m scripts.inspect_data` → `results/data_summary.json`)

| | SMAP | MSL | All |
|---|---|---|---|
| Channels (unique) | 54 | 27 | 81 |
| Labeled anomaly sequences (after merging P-2) | 68 | 36 | 104 |
| point / contextual (label rows, before merging) | 43 / 26 | 19 / 17 | 62 / 43 |
| Train points | 138,004 | 58,317 | 196,321 |
| Test points | 435,826 | 73,729 | 509,555 |
| Fraction of test points labeled anomalous | 12.8% | 10.5% | 12.5% |
| Median of per-channel mean anomaly length (points) | 218 | 131 | 156 |
| Command features | 24 | 54 | — |

Before merging, the counts match the telemanom README (82 rows; 69 SMAP and 36 MSL
sequences). All other rows are computed after merging. Per-channel numbers: `results/tables/data_channels.csv`. Example plots:
`figures/data_examples.png`.

## Things that matter for the evaluation

- **High anomaly density.** 12.5% of test points are labeled anomalous; 13 channels are
  more than 30% anomalous, and three (M-1, M-2, D-2) are about 50%. Real operations have far
  rarer anomalies. Wu & Keogh (2021) flag this density as unrealistic. High density is
  exactly what lets point adjustment reward random scores.
- **Short training series.** Median 2,849 (SMAP) and 2,158 (MSL) training points per
  channel; minimum 312. Validation (last 20%) is therefore only 60–860 points per channel,
  so thresholds are estimated from little data.
- **Some anomalies are trivially visible.** M-6's training series is constant at -1.0,
  and its test series reaches 258. 44 of 82 channels have test values outside the
  training range (by more than 0.05). The benchmark-sanity table measures how many
  anomalies trivial baselines already catch.
- **One sequence per channel, mostly.** 64 of 82 channels have exactly one labeled
  anomaly, so event-wise recall per channel is close to binary.
