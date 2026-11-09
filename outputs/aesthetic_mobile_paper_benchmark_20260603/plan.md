# Mobile Aesthetic Paper Benchmark Plan

Goal: Benchmark NIMA, Mobile A-LAMP v2, RGNet, and ICAA TFLite models against official paper datasets (AVA, AADB, ICAA17K/SPAQ).

## Models & Datasets
- **NIMA** on **AVA**
- **Mobile A-LAMP v2** on **AVA**
- **RGNet** on **AADB**
- **ICAA** on **ICAA17K** (Primary) and **SPAQ** (Secondary/Colorfulness)

## Implementation Details
- Language: Python
- Libraries: `tensorflow-lite`, `numpy`, `PIL`, `pandas`, `scipy` (for SRCC/PLCC)
- Preprocessing:
  - EXIF orientation correction for all.
  - NIMA: 224x224, RGB, normalized /255.
  - A-LAMP v2: 384x384 (full) + 5x 224x224 (patches), RGB, raw 0..255.
  - RGNet: 256x256, PIL resize, RGB, normalized /255.
  - ICAA: 224x224, RGB, normalized /255, then ImageNet mean/std.

## Success Criteria
- Validated inference pipeline for each model.
- Metrics computed: SRCC, PLCC, Accuracy (where applicable).
- Final report comparing results with paper numbers.
