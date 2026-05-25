# TOPIQ-lite ranking experiment context notes

- Scope is experimental only. Existing `src/train/train_topiq_lite.py`, `src/models/topiq_lite.py`, and dataset CSVs remain untouched.
- Regression replay uses `data/processed/topiq_replacement/mixed_112_train.csv` and `mixed_112_val.csv` on normalized MOS.
- Pairwise ranking uses FLIVE train pairs with `abs(score_a - score_b) >= 0.05`, deterministic sampling, and labels in `{-1, +1}`.
- Image preprocessing must match TOPIQ-lite: RGB decode, `resize_with_pad` to 384x384, float32 pixels in `0..255`, and no `/255.0`.
- Smoke validation is bounded to compile checks, pair generation, and the requested 3-epoch run.
- Trainer builds TOPIQ-lite in float32, loads the provided replacement candidate weights, and keeps the backbone frozen when requested.
- Quick FLIVE validation is taken from the FLIVE subset inside the mixed validation CSV because the requested CLI does not provide a separate FLIVE val CSV.
- Deterministic pair generation wrote 10,000 FLIVE pairs to `data/processed/topiq_replacement/flive_pairs_gap005_10k.csv`.
- The first exact smoke attempt exposed CPU-only execution (`Visible GPUs: []`) and would have taken a long full paired epoch, so the trainer now defaults to `--max_steps_per_epoch 100`; set it to `0` or a larger value only for an intentional longer probe.
- Full mixed validation was also too slow on CPU-only execution, so the trainer now defaults to `--max_val_samples 256` and `--quick_flive_limit 256`; set `--max_val_samples 0` and a higher quick limit only for an intentional longer probe.
