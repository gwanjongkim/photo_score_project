# RGNet Mobile-Resize Fine-Tune Preflight

## 1. Summary
This preflight audit designs an experiment to close the confirmed 0.068 SRCC "Resize Gap" in RGNet. By aligning the training-time preprocessing with the mobile deployment environment (PIL-Bilinear), we aim to recover the model's ranking performance on on-device benchmarks.

## 2. Confirmed Resize Gap
Stage 1 alignment checks reproduced the following gap on identical AADB test samples:
- **TF-Native SRCC:** 0.6819
- **PIL / Mobile-like SRCC:** 0.6139
- **Total Gap:** **0.0680 SRCC** (approx. 10% performance drop).

## 3. Current Training Pipeline
- **Script:** `src/train/train_rgnet_paper_v1_aadb.py`
- **Implementation:** Uses `tf.image.decode_jpeg` and `tf.image.resize` inside the `tf.data` map function (`_decode_image`).
- **Interpolation:** Default `tf.image.resize` (Bilinear) which lacks antialiasing and coordinate parity with common CPU-based libraries like PIL.

## 4. Current Evaluation Pipeline
- **Mobile/TFLite Benchmark:** Uses `PIL.Image.resize(BILINEAR)` to simulate Flutter/mobile environment.
- **Normalization:** `pixel_value / 255.0` (range 0..1).

## 5. Checkpoint Availability
- **Best Keras Checkpoint:** `outputs/rgnet_paper_v1_ablation_full_candidates_20260510/full_train/agg_mean_full/final_model.keras`
- **Loadability:** **BLOCKED**. A deserialization mismatch exists. The checkpoint contains a `region_score_activation` parameter in the `RegionScoreAggregation` layer that is missing from the current source code in `src/models/rgnet_paper_v1.py`.
- **Impact:** Fine-tuning from the existing best weights is currently not possible without a code patch.

## 6. Feasibility of Mobile-Like Preprocessing Training
- **PIL Integration:** PIL can be used in the training loop via `tf.py_function`.
- **Training Time:** High. AADB is a relatively small dataset (~8k samples). Previous runs completed in just 4 epochs due to early stopping. Retraining from scratch is computationally inexpensive (~10-20 mins on GPU).

## 7. Candidate Experiments

### Option A: Mobile-like Fine-tune (Resume)
- **Description:** Resume training from the best checkpoint using PIL-Bilinear resize.
- **Benefit:** Faster convergence.
- **Risk:** High (deserialization blocker).
- **Required Changes:** Patch `RegionScoreAggregation` and add `--checkpoint` support to the training script.

### Option B: Mobile-like Retrain from Scratch (Recommended)
- **Description:** Train from ImageNet/backbone init using PIL-Bilinear resize for all epochs.
- **Benefit:** Zero technical debt; guaranteed preprocessing parity.
- **Risk:** Low.
- **Output Dir:** `outputs/rgnet_v1_aadb_pil_resize_retrain_20260524`

### Option C: Multi-style Resize Augmentation
- **Description:** Randomly switch between `tf.image.resize` and `PIL.Image.resize` during training.
- **Benefit:** Increased robustness to different deployment backends.
- **Risk:** May result in slightly lower peak performance on a single target.

## 8. Recommended First Experiment
**Option B: Mobile-like Retrain from Scratch**.
Given the small size of AADB and the code-checkpoint mismatch, retraining from scratch with PIL-Bilinear resize is the fastest and safest path to recovering the 0.07 SRCC gap.

## 9. Success Criteria
- **Primary:** AADB mobile-like SRCC improves from 0.6139 to **>= 0.66**.
- **Strong:** AADB mobile-like SRCC approaches **0.68** (full recovery).
- **Paper-near:** AADB mobile-like SRCC **>= 0.70** (close to 0.7104 target).

## 10. Required Code Changes (via Codex)
1. Modify `src/train/train_rgnet_paper_v1_aadb.py` to support PIL-Bilinear resize.
2. Use `tf.py_function` in `_decode_image` to wrap the PIL resizing logic.
3. Ensure normalization remains `[0, 1]` for RGNet.

## 11. Risks
- **IO Bottleneck:** `tf.py_function` with PIL is slower than native TF. However, for AADB's size, this should be negligible.
- **Paper SOTA:** Retraining with resize parity will likely reach ~0.68, but reaching the 0.7104 paper target may still require the Stage 2 "Faithful Teacher" architecture.

## 12. Final Recommendation
Initiate a Codex task to implement PIL-Bilinear resizing in the RGNet training script, then run a full retrain on AADB. This will eliminate the Resize Gap as a performance bottleneck.
