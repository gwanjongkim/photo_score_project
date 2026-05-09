# RGNet Paper-Oriented v1 AADB Regression Context Notes

## 2026-05-09

- Existing v0 files are present and left untouched: `src/models/rgnet_paper.py`, `src/train/train_rgnet_paper_aadb.py`, `src/eval/evaluate_rgnet_paper_aadb.py`, and `configs/paper_benchmarks/rgnet_paper_aadb_regression.yaml`.
- Existing practical RGNet `src/models/rgnet.py` is untracked in this working tree and must not be modified.
- Existing `forWeights/` is untracked and must not be used.
- AADB counts from current CSVs: train `7612`, val `846`, test `1000`.
- AADB score ranges: train `0.0..1.0`, val `0.0..0.95`, test `0.05..1.0`.
- v0 full run local artifact: `outputs/rgnet_paper_aadb_regression_20260509/full_train/train_summary.json`, 5 epochs completed, best epoch `2`, best val_loss `0.022423287853598595`, best val_mae `0.11850318312644958`.
- v0 full evaluation local artifact: `outputs/rgnet_paper_aadb_regression_20260509/eval/evaluation_summary.json`, test SRCC `0.6432537443293759`, PLCC `0.6499109307164675`, MAE `0.1255093365907669`, RMSE `0.1556738782154776`.
- Safer implementation choice: add separate v1 model/train/eval files rather than extending v0 defaults.
- Implemented v1 in separate files: `src/models/rgnet_paper_v1.py`, `src/train/train_rgnet_paper_v1_aadb.py`, and `src/eval/evaluate_rgnet_paper_v1_aadb.py`.
- v1 architecture includes DenseNet121, ASPP approximation with dilation rates `[1,3,6,12,18]`, spatial post-ASPP region nodes, cosine softmax adjacency, default `3` residual graph blocks, per-region sigmoid scores, and configurable `mean`/`max`/`lse` aggregation with default LSE `r=4`.
- Pure forward smoke passed on random input `[2,256,256,3]`: output `[2,1]`, finite, in `[0,1]`, prediction range `0.35727399587631226..0.3581896424293518`.
- First smoke train passed but showed a Keras warning for missing `ASPPContextModule.build()`. Added explicit `build()` and reran into `smoke_train_v2/`.
- Verified smoke train output is `smoke_train_v2/`: 16 train samples, 8 val samples, 1 epoch, random DenseNet weights, val_loss `0.17236243188381195`, val_mae `0.3674493432044983`, save/load max abs diff `0.0`.
- Verified smoke eval output is `smoke_eval/`: test subset 8 samples SRCC `-0.15569141404872364`, PLCC `-0.18820356670295135`, MAE `0.4321957230567932`, RMSE `0.45712923157078755`; val subset 8 samples SRCC `-0.8263621207201486`, PLCC `-0.6750406450443565`, MAE `0.3674493730068207`, RMSE `0.4151655475636339`.
- TensorFlow inside the sandbox did not expose CUDA for forward/smoke runs; `nvidia-smi` does show an RTX 4070 SUPER, so mid/full training should use escalated GPU access.
- Mid train completed with escalated GPU access: 512 train samples, 128 val samples, 3 epochs, ImageNet weights, batch size 8. GPU was visible as RTX 4070 SUPER. Validation losses were `0.036103565245866776`, `0.035932641476392746`, `0.03285215049982071`; best val_mae `0.14118336141109467`; save/load diff `0.0`.
- Mid train logged TensorFlow GPU memory-pressure warnings at batch size 8, but no fatal OOM.
- Full train completed with escalated GPU access: 7612 train samples, 846 val samples, ImageNet weights, batch size 8, requested 20 epochs, early-stopped after 5 epochs. Best epoch `2`, best val_loss `0.022888537496328354`, best val_mae `0.11879914253950119`; save/load diff `0.0`.
- Full train history shows overfitting after epoch 2: train loss continued improving from `0.02274704910814762` to `0.015947626903653145`, while val_loss worsened from `0.022888537496328354` to `0.02393358387053013`.
- Full eval completed on test and val: test SRCC `0.638855657699948`, PLCC `0.6423166841140971`, MAE `0.12822051346302032`, RMSE `0.16023966315074326`, seconds/image `0.023911475160999545`; val SRCC `0.5832142747791366`, PLCC `0.5873551037219259`, MAE `0.11880123615264893`, RMSE `0.15128684218806654`, seconds/image `0.013293206747046961`.
- v1 did not beat v0 on the full AADB test split: v0 test SRCC `0.6432537443293759`, v1 test SRCC `0.638855657699948`.
