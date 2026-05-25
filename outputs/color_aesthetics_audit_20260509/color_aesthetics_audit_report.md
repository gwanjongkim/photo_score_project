# Color Aesthetics Feasibility Audit

## 1. Environment
- CWD: `/home/omen_pc1/photo_score_project`
- Date: 2026-05-09
- System has active deployments and evaluations for NIMA, RGNet, A-LAMP, and MUSIQ.
- RGNet-paper-v1 track is currently actively being evaluated and should not be disturbed.

## 2. Paper and Official Resources
- **Paper**: "Thinking Image Color Aesthetics Assessment: Models, Datasets and Benchmarks"
- **Venue**: ICCV 2023
- **Task**: Image Color Aesthetics Assessment (ICAA)
- **Datasets**: ICAA17K, SPAQ
- **Model**: Delegate Transformer (DAT) with SoftHistogram
- **Metrics**: SRCC, PLCC, Accuracy
- **Official Code**: Available via GitHub (`https://github.com/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment`). Cloned locally to `external/icaa_official_repo`.
- **Pretrained Weights**: Available via Google Drive links in the official repository README.
- **Dataset Download**: Available via Google Drive/Baidu Pan links.
- **License**: Apache 2.0

## 3. Local Project Inventory
- **Existing Models**: There are no existing local color aesthetics models in `src/models/` (only NIMA, RGNet, A-LAMP, MUSIQ, and technical quality regressors).
- **Existing Integration**: No existing tracks in the Flutter app or `forWeights` currently factor in dedicated color aesthetics beyond what general models naturally perceive.
- **Documentation**: We have previously generated an audit of the official repo in `docs/icaa_official_repo_dataset_audit.md`.

## 4. Dataset Availability
- **AVA**: Active local availability (`data/processed/ava`, `data/raw/ava/images`).
- **AADB**: Active local availability (`data/processed/aadb`, `data/raw/aadb/images`).
- **ICAA17K**: Downloaded as a ZIP file at `/mnt/c/Users/OMEN PC1/Downloads/ICKK17K_dataset.zip` (~2.6GB). Not yet extracted to the local project's `data/` directory.
- **SPAQ**: Available at `/home/omen_pc1/dataset_zips/spaq_extracted/spaq`. Not directly symlinked into `data/raw/spaq` yet.

## 5. Possible Tracks
**Track D0: Hand-crafted color aesthetics baseline**
- *Requirements*: Basic OpenCV/scikit-image scripts for palette, hue variance, colorfulness metrics.
- *Difficulty*: Low.
- *Expected Output*: A Python script to append a hand-crafted `color_score` to any CSV manifest.
- *Parallelism*: Fully parallel; does not touch deep learning models.
- *Mobile Suitability*: Very high (can be implemented in pure Dart/C++ on mobile).

**Track D1: Official ICAA inference wrapper**
- *Requirements*: Isolated environment (`.venv_icaa` with PyTorch), downloaded pretrained weights.
- *Difficulty*: Medium (requires bridging legacy PyTorch code).
- *Expected Output*: Inference script logging ICCV scores.
- *Parallelism*: High. Kept in `external/` or `src/color_aesthetics/`.
- *Mobile Suitability*: **Very Low**. The official model relies heavily on `F.grid_sample` which breaks TFLite conversion easily.

**Track D2: ICAA training/retraining**
- *Requirements*: Extracting the 2.6GB ICAA17K zip to `data/raw/icaa17k`, constructing PyTorch dataset loaders.
- *Difficulty*: High (requires adapting the NNI-based training loop to local infrastructure).
- *Expected Output*: Locally trained model weights.
- *Parallelism*: High (entirely distinct data and architecture).
- *Mobile Suitability*: Low (same TFLite constraints as D1).

**Track D3: forWeights integration**
- *Requirements*: A mobile-friendly color model (e.g., distilling the official model into a MobileNet, or using D0).
- *Difficulty*: Medium-High.
- *Expected Output*: An added TFLite artifact for the `forWeights` evaluation tool.
- *Mobile Suitability*: Requires a distillation track or fallback to D0.

**Track D4: Flutter integration later**
- Dependent on resolving D3.

## 6. Relationship to RGNet/A-LAMP/NIMA
A color aesthetics track is entirely orthogonal to existing aesthetic models.
It does not conflict with `RGNet-paper-v1`, which focuses on composition/region graphs, nor does it share datasets with NIMA/A-LAMP (which use AVA/AADB). This means it can be pursued entirely in parallel.

## 7. Recommended Implementation Order
1. **Safety First**: Do not disturb `RGNet-paper-v1`.
2. **Setup Isolation**: Create isolated folders:
   - `src/color_aesthetics/`
   - `tools/color_aesthetics/`
   - `configs/color_aesthetics/`
   - `outputs/color_aesthetics_audit_YYYYMMDD/`
3. **Execute Track D0**: Establish a lightweight hand-crafted baseline first.
4. **Execute Track D1**: Set up a server-side PyTorch wrapper to evaluate the official weights over our datasets as an oracle color metric.
5. **Evaluate Distillation (for D3/D4)**: If D1 provides high value, consider knowledge distillation to a MobileNetV2 architecture to bypass the `F.grid_sample` TFLite blocker.

## 8. Risks
- **Dependency Hell**: The official code requires PyTorch 1.7.1, which conflicts with our TensorFlow-centric workflow and modern Python requirements.
- **TFLite Export Blockers**: The `Delegate Transformer` architecture is not natively mobile-friendly.
- **Dataset Storage**: Expanding ICAA17K and SPAQ will consume further disk space and require data preparation scripts parallel to our `make_ava_csv.py` workflows.

## 9. Next Codex Prompt
"Set up the Track D0 hand-crafted color aesthetics baseline. Create an isolated folder `src/color_aesthetics/`. Implement a Python script `src/color_aesthetics/handcrafted_baseline.py` using OpenCV to calculate color harmony, colorfulness, and dominant hue statistics from an image, outputting a scalar score. Provide a smoke test using `test_samples/`."
