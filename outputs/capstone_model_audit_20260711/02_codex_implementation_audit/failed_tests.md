# Failed tests

| Command/test | Failure | Consequence |
|---|---|---|
| `test -d outputs/.../01_paper_baseline` | exit 1; directory absent | Used explicitly available `01_paper_baseline_codex`; baseline path mismatch remains. |
| `python -c ... production_technical_predictions.csv` | `/bin/bash: python: command not found` | Replaced later execution with `./.venv_gpu/bin/python`; failure not concealed. |
| `nl -ba src/models/icaa_dat.py` | `No such file or directory` | Local ICAA implementation/head semantics not verified. |
| Independent metric recomputation | no ground-truth column in 4,967-row technical predictions table | No SRCC/PLCC/MAE/RMSE positive verdict. |

No TFLite invocation in the bounded seven-model set failed.
