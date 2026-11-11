| spacecraft   | method            |   seeds | event P       | event R       | event F1      | point F1 (no adjust)   | PA-F1 (inflated)   | false alarms / 1k steps   |
|:-------------|:------------------|--------:|:--------------|:--------------|:--------------|:-----------------------|:-------------------|:--------------------------|
| SMAP         | Tiny MLP, INT8    |       3 | 0.763 ± 0.019 | 0.740 ± 0.017 | 0.751 ± 0.015 | 0.326 ± 0.008          | 0.900 ± 0.014      | 0.04 ± 0.00               |
| SMAP         | Tiny MLP, float32 |       3 | 0.788 ± 0.020 | 0.745 ± 0.008 | 0.766 ± 0.013 | 0.294 ± 0.009          | 0.876 ± 0.017      | 0.03 ± 0.00               |
| SMAP         | Persistence       |       1 | 0.850         | 0.500         | 0.630         | 0.153                  | 0.426              | 0.01                      |
| MSL          | Tiny MLP, INT8    |       3 | 0.660 ± 0.033 | 0.481 ± 0.042 | 0.556 ± 0.025 | 0.269 ± 0.013          | 0.359 ± 0.027      | 0.12 ± 0.02               |
| MSL          | Tiny MLP, float32 |       3 | 0.686 ± 0.039 | 0.481 ± 0.032 | 0.565 ± 0.024 | 0.270 ± 0.012          | 0.355 ± 0.023      | 0.11 ± 0.02               |
| MSL          | Persistence       |       1 | 0.750         | 0.417         | 0.536         | 0.231                  | 0.335              | 0.07                      |
| ALL          | Tiny MLP, INT8    |       3 | 0.734 ± 0.024 | 0.651 ± 0.006 | 0.689 ± 0.008 | 0.317 ± 0.008          | 0.838 ± 0.016      | 0.05 ± 0.01               |
| ALL          | Tiny MLP, float32 |       3 | 0.758 ± 0.003 | 0.654 ± 0.010 | 0.702 ± 0.005 | 0.290 ± 0.007          | 0.816 ± 0.016      | 0.04 ± 0.00               |
| ALL          | Persistence       |       1 | 0.817         | 0.471         | 0.598         | 0.166                  | 0.412              | 0.02                      |

Threshold: telemanom, fitted on the validation split (end of train) only. Event-wise = telemanom definition; PA-F1 shown only for comparison with random scores.
