# Stage A Checklist

- [x] Create Stage A directories.
- [x] Add `inspect_pytorch_state_dict.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/inspect_pytorch_state_dict.py`.
- [x] Verify CSV inventory exists and is complete.
- [x] Verify JSON summary exists.
- [x] Verify Markdown report exists.
- [x] Confirm no official repo, A-cut, Flutter, training, or conversion changes were made.

# Stage B Checklist

- [x] Create Stage B directories.
- [x] Add `stage_b_basic_layer_parity.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/stage_b_basic_layer_parity.py`.
- [x] Verify Stage B JSON report exists.
- [x] Verify Stage B log exists.
- [x] Verify Stage B Markdown report exists.
- [x] Confirm no DAT implementation, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage C Checklist

- [x] Create Stage C directories.
- [x] Add `tf_models/patch_projection.py`.
- [x] Add `stage_c_patch_projection_parity.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/stage_c_patch_projection_parity.py`.
- [x] Verify Stage C JSON report exists.
- [x] Verify Stage C log exists.
- [x] Verify Stage C Markdown report exists.
- [x] Confirm no later DAT stages, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage D Checklist

- [x] Inspect PyTorch `SoftHistogram.py`.
- [x] Create Stage D directories.
- [x] Add `tf_models/soft_histogram.py`.
- [x] Add `stage_d_soft_histogram_parity.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/stage_d_soft_histogram_parity.py`.
- [x] Verify Stage D JSON report exists.
- [x] Verify Stage D log exists.
- [x] Verify captured tensor summary exists.
- [x] Verify Stage D Markdown report exists.
- [x] Confirm no full DAT implementation, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage E Checklist

- [x] Inspect PyTorch `dat.py` and `dat_blocks.py`.
- [x] Identify stage-0 depth-0 class names and checkpoint tensors.
- [x] Add `tf_models/local_attention_block.py`.
- [x] Add `stage_e_local_attention_block_parity.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/stage_e_local_attention_block_parity.py`.
- [x] Verify Stage E JSON report exists.
- [x] Verify Stage E log exists.
- [x] Verify debug tensor summary exists.
- [x] Verify Stage E Markdown report exists.
- [x] Confirm no deformable attention, full DAT implementation, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage F Checklist

- [x] Inspect PyTorch `DAttentionBaseline` implementation.
- [x] Identify first `DAttentionBaseline` instance and checkpoint tensors.
- [x] Add `tf_models/deformable_attention.py`.
- [x] Add `stage_f_deformable_attention_parity.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/stage_f_deformable_attention_parity.py`.
- [x] Verify Stage F JSON report exists.
- [x] Verify Stage F log exists.
- [x] Verify debug tensor summary exists.
- [x] Verify Stage F Markdown report exists.
- [x] Confirm no full DAT implementation, full TransformerStage implementation, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage G-0 Checklist

- [x] Inspect PyTorch `TransformerStage` stage `0` structure and checkpoint tensor names.
- [x] Add `tf_models/transformer_stage.py` for stage `0` only.
- [x] Add `stage_g0_transformer_stage0_parity.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/stage_g0_transformer_stage0_parity.py`.
- [x] Verify Stage G-0 JSON report exists.
- [x] Verify Stage G-0 log exists.
- [x] Verify debug tensor summary exists.
- [x] Verify Stage G-0 Markdown report exists.
- [x] Confirm no full DAT implementation, later stage implementation, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage G-1 Checklist

- [x] Inspect PyTorch `TransformerStage` stage `1` structure and checkpoint tensor names.
- [x] Update `tf_models/transformer_stage.py` for stage `1` only.
- [x] Add `stage_g1_transformer_stage1_parity.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/stage_g1_transformer_stage1_parity.py`.
- [x] Verify Stage G-1 JSON report exists.
- [x] Verify Stage G-1 log exists.
- [x] Verify debug tensor summary exists.
- [x] Verify Stage G-1 Markdown report exists.
- [x] Confirm no full DAT implementation, stage `2` or `3` implementation, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage G-2 Checklist

- [x] Inspect PyTorch `TransformerStage` stage `2` structure and checkpoint tensor names.
- [x] Update `tf_models/transformer_stage.py` for stage `2` only.
- [x] Add `stage_g2_transformer_stage2_parity.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/stage_g2_transformer_stage2_parity.py`.
- [x] Verify Stage G-2 JSON report exists.
- [x] Verify Stage G-2 log exists.
- [x] Verify debug tensor summary exists with per-block residual diffs.
- [x] Verify Stage G-2 Markdown report exists.
- [x] Confirm no full DAT implementation, stage `3` implementation, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage G-2 MLP Surgical Debug Checklist

- [x] Add `debug_stage_g2_block_mlp.py`.
- [x] Capture block `3` and block `4` full-flow subcomponents.
- [x] Run direct same-input PyTorch-captured `norm2` MLP parity for block `3` and block `4`.
- [x] Run `python experiments/icaa_tf_native/scripts/debug_stage_g2_block_mlp.py`.
- [x] Verify debug JSON report exists.
- [x] Verify debug log exists.
- [x] Verify optional tensor summary exists.
- [x] Verify Markdown debug report exists.
- [x] Confirm no model-code fix, full DAT implementation, stage `3`, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage G-3 TransformerStage 3 Checklist

- [x] Inspect PyTorch `TransformerStage` stage `3` structure and `stages.3.*` checkpoint keys.
- [x] Add `TFTransformerStage3` only.
- [x] Add `stage_g3_transformer_stage3_parity.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/stage_g3_transformer_stage3_parity.py`.
- [x] Verify Stage G-3 JSON report exists.
- [x] Verify Stage G-3 log exists.
- [x] Verify optional debug tensor summary exists.
- [x] Verify Stage G-3 Markdown parity report exists.
- [x] Confirm no full DAT implementation, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage G-3 Block1 DAttention Surgical Debug Checklist

- [x] Add `debug_stage_g3_block1_dattention.py`.
- [x] Capture PyTorch Stage 3 block `1` norm-attention tensor and feed it directly to PyTorch and TensorFlow DAttention.
- [x] Compare DAttention subcomponents from input through final output projection.
- [x] Run `python experiments/icaa_tf_native/scripts/debug_stage_g3_block1_dattention.py`.
- [x] Verify debug JSON report exists.
- [x] Verify debug log exists.
- [x] Verify optional debug tensor summary exists.
- [x] Verify Markdown debug report exists.
- [x] Confirm no model-code fix, full DAT implementation, training, conversion, official repo edits, A-cut edits, or Flutter edits were made.

# Stage H Full DAT Parity Checklist

- [x] Confirm PyTorch `DAT.forward` order and full-model checkpoint groups.
- [x] Add inference-only `tf_dat_model.py`.
- [x] Add `stage_h_full_dat_parity.py`.
- [x] Run `python experiments/icaa_tf_native/scripts/stage_h_full_dat_parity.py`.
- [x] Verify Stage H JSON report exists.
- [x] Verify Stage H predictions CSV exists.
- [x] Verify Stage H log exists.
- [x] Verify optional intermediate diff summary exists.
- [x] Verify Stage H Markdown parity report exists.
- [x] Confirm no SavedModel export, TFLite conversion, training, deployment code, official repo edits, A-cut edits, or Flutter edits were made.
