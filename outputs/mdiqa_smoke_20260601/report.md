# MDIQA WSL Smoke Test Report

## Scope
- Smoke test only.
- No training, TFLite export, Flutter edits, production scoring edits, distillation, or teacher-label creation.
- License risk: S-Lab Non-commercial / Academic-only.

## Environment
- environment_path: `/home/omen_pc1/photo_score_project/.venv_mdiqa`
- repo_path: `/home/omen_pc1/photo_score_project/MDIQA`
- open_clip_path: `/home/omen_pc1/photo_score_project/open_clip`
- device: `cpu` unless CUDA is visible to PyTorch.

## Checkpoints
- fusion: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/mdiqa_shared_5t_4a_mbv3_koniq_seed1.pth` (present, 19930987 bytes)
- technology_backbone: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_freeze_backbone.pth` (present, 144754849 bytes)
- technology_head0: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_koniq10k_val_head0_head0.pth` (present, 36428809 bytes)
- technology_head1: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_koniq10k_val_head1_head1.pth` (present, 36428809 bytes)
- technology_head2: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_koniq10k_val_head2_head2.pth` (present, 36428809 bytes)
- technology_head3: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_koniq10k_val_head3_head3.pth` (present, 36428809 bytes)
- technology_head4: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_koniq10k_val_head4_head4.pth` (present, 36428809 bytes)
- aesthetic_color: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_PARA_val_color.pth` (present, 290502209 bytes)

## Images
- /home/omen_pc1/photo_score_project/test_samples/KakaoTalk_20260330_180646779.jpg
- /home/omen_pc1/photo_score_project/test_samples/KakaoTalk_20260330_180646779_01.jpg
- /home/omen_pc1/photo_score_project/test_samples/KakaoTalk_20260330_180646779_02.jpg
- /home/omen_pc1/photo_score_project/test_samples/KakaoTalk_20260330_180646779_03.jpg
- /home/omen_pc1/photo_score_project/test_samples/KakaoTalk_20260330_180646779_04.jpg

## Raw Outputs
- KakaoTalk_20260330_180646779.jpg: 0.5987405180931091
- KakaoTalk_20260330_180646779_01.jpg: 0.5871518850326538
- KakaoTalk_20260330_180646779_02.jpg: 0.5902355909347534
- KakaoTalk_20260330_180646779_03.jpg: 0.5416154861450195
- KakaoTalk_20260330_180646779_04.jpg: 0.5722495913505554

## Output Scale
- likely 0..1

## Failures
- None.

## Final Decision
MDIQA_READY_FOR_100_IMAGE_SANITY
