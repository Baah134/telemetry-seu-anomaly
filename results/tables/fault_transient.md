| model                |   upset prob. per inference | event F1 mean ± std   |   false alarms / 1k | catastrophic trials   | trials with NaN/inf   |
|:---------------------|----------------------------:|:----------------------|--------------------:|:----------------------|:----------------------|
| float32              |                      0.0001 | 0.500 ± 0.064         |               0.057 | 0%                    | 0%                    |
| float32              |                      0.001  | 0.453 ± 0.097         |               0.097 | 0%                    | 50%                   |
| float32              |                      0.01   | 0.291 ± 0.203         |               0.138 | 50%                   | 100%                  |
| float32              |                      0.1    | 0.119 ± 0.082         |               0.3   | 100%                  | 100%                  |
| INT8 (integer model) |                      0.0001 | 0.615 ± 0.000         |               0.081 | 0%                    | 0%                    |
| INT8 (integer model) |                      0.001  | 0.615 ± 0.000         |               0.081 | 0%                    | 0%                    |
| INT8 (integer model) |                      0.01   | 0.711 ± 0.022         |               0.024 | 0%                    | 0%                    |
| INT8 (integer model) |                      0.1    | 0.644 ± 0.073         |               0.065 | 0%                    | 0%                    |

One random bit flip in a random int32 accumulator (INT8) or float32 pre-activation (float) with the given probability per inference.
