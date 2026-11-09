# Executable tests

Command: `./.venv_gpu/bin/python outputs/.../run_bounded_checks.py`. Seed: 20260711. CPU TensorFlow Lite XNNPACK. Full raw output is `bounded_checks.json`.

| Test | Result | Direct runtime evidence |
|---|---|---|
| Core manifests load | PASS | KonIQ 10,073; FLIVE image 39,810; AADB 9,458 rows across splits |
| Exact path split duplicates | PASS for tested groups | zero for all train/val/test pairs |
| Missing image paths | PASS for tested groups | zero in all nine manifests |
| TFLite load/allocate | PASS | seven selected models allocated successfully |
| Deterministic forward pass | PASS | all seven invoked with seeded input |
| Output NaN/Inf | PASS | all outputs finite |
| NIMA output shape | PASS | `[1,10]` |
| NIMA distribution sum | PASS | exactly 1.0 for executed input |
| Scalar output shapes | PASS | KonIQ, FLIVE, RGNet, both A-LAMP variants, Stage5 student: `[1,1]` |
| Accuracy/correlation | NOT EXECUTED | selected prediction CSV lacks ground truth |
| Checkpoint loading | UNVERIFIED | bounded test loaded TFLite, not every Keras/PyTorch checkpoint |
| ICAA local construction | FAILED/NOT FOUND | `src/models/icaa_dat.py` absent |
| Full MDIQA teacher forward | NOT EXECUTED | dependency/checkpoint cost outside bounded pass |

Random inputs validated interface stability only. They were not necessarily semantically correct inputs; notably Mobile A-LAMP v2 expects `[0,255]`, while the generic test generated `[0,1]`.
