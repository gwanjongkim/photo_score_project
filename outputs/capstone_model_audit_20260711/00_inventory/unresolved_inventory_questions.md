# Unresolved inventory questions

Questions are ordered by their impact on later fidelity and reproducibility work. This phase does not answer them by inventing a fallback.

## P0. Runtime and production identity

1. Which exact Flutter checkout, commit, and build artifact is the submitted or currently active app?
2. Are `exports/tflite/koniq_mobile.tflite` and `flive_image_mobile.tflite` byte-identical to the active app assets?
3. Which technical policy is authoritative in the app: default Stage5 five-signal weights, image-only 4/7-3/7, the 0.60/0.40 helper, or another Dart implementation?
4. Which aesthetic implementation is authoritative: the default old Stage5 server ensemble, the Stage 5 student, the June calibrated three-model ensemble, or external Flutter weights?
5. Why does the default server config still load `checkpoints/nima_ava_gpu/final_model.keras` after the preprocessing-fix report classifies it as defective?
6. Are fixed NIMA, PIL-resize RGNet, and Mobile A-LAMP v2 present in the active Flutter asset manifest and actually invoked?
7. Does any active runtime load ICAA? If so, what exact preprocessing, output index, and score meaning does it use?
8. Was TOPIQ ever active by default, or only shadow/log-only? What exact feature flag and commit prove that state?
9. Are Stage 5 student and MDIQA student assets absent from the active app, as later reports claim?
10. Which exact mobile runtime file implements the final technical and aesthetic score constants?

## P0. ICAA contract conflicts

11. Why do `outputs/icaa_performance_reverification_20260525/report.md` and other reports call the FP16 file about 11.9 MB when the current file is 175,885,416 bytes?
12. Why does the validated export path apply ImageNet normalization while `forWeights2/configs/aesthetic_weight_lab.yaml` applies only zero-to-one normalization?
13. Why does the validated color path use output index 1 while the weight-lab config uses index 0/MOS?
14. Is there a missing wrapper model that embeds ImageNet preprocessing and/or reorders outputs?
15. What upstream release, commit, and download source produced `weights/icaa_official/e_30_...pth`?
16. Which actual revision and command trained the official ICAA checkpoint? The current `main_nni.py` training call is commented and the configured trial script is missing.

## P1. Original training provenance

17. What were the exact commands, environment, seeds, target columns, and stdout logs for `technical_koniq_gpu`, `technical_flive_image_gpu`, and `technical_flive_patch_gpu`?
18. Did the production KonIQ checkpoint train against raw MOS or `MOS_zscore`? The current builder appears to select the latter.
19. What were the exact original commands and resolved configs for practical NIMA, RGNet, A-LAMP, AADB composition, MUSIQ AADB, and pairwise checkpoints?
20. What exact AVA raw annotation file and invocation produced the current processed AVA CSVs? The builder's expected annotation source is absent.
21. What script produced the fixed NIMA 2,000-image metrics and `nima_prediction_diagnostics.csv`?
22. What exact U2Net saliency and V4 patch-selection commands created the Mobile A-LAMP full JSONL manifests?
23. Why does the Mobile A-LAMP report retain “full 4096 training not run” wording while summary/log/model artifacts record a full run?
24. What exact training command created TOPIQ mixed112? The saved command uses `--init_weights`, absent from the current trainer.
25. What exact commands trained and exported the three local MUSIQ technical probes?
26. What exact command trained `checkpoints/musiq_aadb_gpu`, and which target column did it use?
27. What was the original invocation for `checkpoints/student_full_aadb`, and where is its complete stdout log?
28. Which A-LAMP teacher branches were used for actual teacher labels versus only benchmarked?

## P1. Dataset and split provenance

29. Is the current KonIQ 80/10/10 split the actual production training split, and is its label scale intentional?
30. Which FLIVE/PaQ-2-PiQ split is authoritative for each claim: the March custom split, toolbox-standardized split, or another official split?
31. The repository did not recover an original-author PaQ-2-PiQ official split. Which later claims depend on calling the local split “official”?
32. How do MDIQA official config split files map to the repository's audit benchmark manifest?
33. Are the 14,105-row student labels correctly described as a validated cache subset rather than official full-dataset relabeling everywhere?
34. Can every 50k teacher row be traced to its exact MDIQA checkpoint variant and inference version?
35. Were AVA/AADB aligned prediction tables produced from the exact TFLite candidate bytes now in `forWeights2`?

## P1. Checkpoint and export provenance

36. The MDIQA checkpoint hashes were recomputed, but what immutable upstream release or Drive metadata maps each file to its claimed variant?
37. Why do KonIQ/FLIVE verification JSON files predate the current TFLite mtimes? Were the files recopied without modification or re-exported?
38. Which checkpoint produced each duplicated NIMA/RGNet/A-LAMP TFLite in `models/`, `exports/`, `forWeights/`, and `forWeights2/`?
39. Is the RGNet PIL-resize `forWeights2` file copied directly from the audited export? Current SHA confirms equality, but the copy command is absent.
40. Is the Mobile A-LAMP `forWeights2` file copied directly from the parity-verified export? Add the missing immutable source metadata.
41. Why is there no TFLite for FLIVE patch, TechIQA-Guard, DistortionGuard, or the 50k MDIQA student? Some are intentional no-go outcomes; document that per artifact.
42. Which SavedModel variable files should be included in later full provenance hashing in addition to `saved_model.pb`?

## P2. Claims and chronology

43. Which report is authoritative for submitted aesthetic weights: 0.40/0.45/0.15, 0.10/0.50/0.40, or the offline 0.37/0.41/0.22 recommendation?
44. Which contract names are authoritative for the submitted A-LAMP and RGNet assets?
45. Why does an advisor weekly report say TOPIQ was finally adopted while later decision reports say log-only?
46. When exactly did Stage 5 student change from app candidate to rejected ablation, and is the handoff clearly marked historical?
47. Are the production labels in `final_model_status.csv` implementation facts, thesis recommendations, or desired future state?
48. Can the manual Mac supplemental verification be backed by an immutable commit, asset hashes, and captured command output?
49. Which latency claims come from reproducible device logs, and which are representative values copied into reports?
50. Which paper/baseline metrics use official versus custom/random splits?

## P2. Repository and environment

51. Many important source/checkpoint/output files are untracked or ignored. Which files and hashes form the minimum reproducibility package?
52. Is `.venv_mdiqa` intentionally on Torch 2.12.0 while `.venv_gpu` is Torch 2.5.1+cu121 and the WSL driver reports CUDA 13.2 compatibility?
53. Which environment produced each training/export run? Resolved package lockfiles are generally absent.
54. Does the lack of `nvcc` matter for any claimed local CUDA build, custom op, or exporter?
55. The working tree began dirty with 14 tracked modifications and 8,965 untracked file entries. Which state should later fidelity auditing treat as the capstone submission baseline?
56. Are current modified A-LAMP trainer/evaluator/config files the versions that produced existing checkpoints?
57. What commit or archive corresponds to the actual paper/report results rather than the current ahead-of-origin working tree?
58. Which vendored `MDIQA/pyiqa` architectures were actually exercised by the capstone, beyond MDIQA/TOPIQ/MUSIQ wrappers?

## Suggested evidence needed in the next phase

- Exact Flutter repository path, commit, `pubspec.yaml`, Dart contracts, weights constants, and SHA-256 of bundled assets.
- Per-run command, resolved config, Git SHA, environment lock, manifest hash, checkpoint hash, export hash, and evaluation hash.
- A single immutable mapping table from every report claim to raw metric/prediction rows and the code version that produced them.
- Clarified historical/superseded markers on handoff and weekly-report claims.

