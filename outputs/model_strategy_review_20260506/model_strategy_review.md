# A-cut Model Strategy Review

## 1. Investigation Environment

- WSL path: `/home/omen_pc1/photo_score_project`
- Flutter path inspected: `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app`
- Flutter branch: `feat/acut`
- Flutter status: dirty working tree, `git status --short | wc -l` returned `626`
- Flutter HEAD: `8d31cb394fb83baaf43ed84582b0068ae9c1b20e`
- Python: system `Python 3.12.3`; project venv `./.venv_gpu/bin/python` also `Python 3.12.3`
- GPU availability: `nvidia-smi` reported NVIDIA-SMI `590.48.01`, CUDA `13.1`, and one NVIDIA GeForce RTX 4070-class GPU
- Commands used: environment probes, `git`, `find`, `rg`, `head`, `wc`, `awk`, `readlink`, `ls`, `nl -ba`, targeted model/script/metadata inspection, and report-file creation. Full command notes are in `outputs/model_strategy_review_20260506/command_log.txt`.
- Inaccessible paths: none observed for the inspected WSL path and Flutter path. Exact macOS path `/Users/gwanjung_mac/StudioProjects/pozy_app` was not inspected because the expected local Flutter repository was available under `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app`.

No source code was modified and no full training was started.

## 2. Direct Answers

### Q1. Are NIMA and RGNet the same as the papers?

NIMA: close implementation.

Local evidence supports the core NIMA structure: 10 score buckets, softmax output, CDF/EMD-style loss, AVA rating distributions, ImageNet-pretrained CNN backbone, and expectation over bins 1..10. Evidence: `/home/omen_pc1/photo_score_project/src/models/nima_distribution.py:1`, `:27`, `:37`, `:43`, `:53`; `/home/omen_pc1/photo_score_project/src/train/train_nima.py:42`, `:75`; `/home/omen_pc1/photo_score_project/src/datasets/ava_distribution_dataset.py:65`, `:88`; `/home/omen_pc1/photo_score_project/exports/tflite/nima_mobile.metadata.json:15`; `/home/omen_pc1/photo_score_project/exports/tflite/nima_mobile.verify.json:6`.

It is not an exact paper reproduction from local evidence because the implementation explicitly uses EfficientNetV2B0 as the backbone and documents that as an approximation: `/home/omen_pc1/photo_score_project/src/models/nima_distribution.py:19`, `:27`. The local evidence does not prove exact paper preprocessing, exact original backbone choice, or exact original training protocol.

RGNet: practical RGNet-style approximation.

Local evidence supports graph reasoning: region feature projection, feature-similarity adjacency, graph convolution, region weighting, scalar regression, and TFLite graph-like ops from the previous local audit log. Evidence: `/home/omen_pc1/photo_score_project/src/models/rgnet.py:55`, `:76`, `:126`; `/home/omen_pc1/photo_score_project/exports/tflite/rgnet_aadb_gpu.verify.json:6`; `/home/omen_pc1/photo_score_project/outputs/audit_model_paper_basis_20260506/command_log.txt:305`.

It is not an exact paper reproduction because the current model uses EfficientNetV2B0, not DenseNet-121 or an FCN/DenseASPP paper encoder. Evidence: `/home/omen_pc1/photo_score_project/src/models/rgnet.py:19`, `:32`. DenseASPP or the original exact RGNet encoder was not verifiable from local evidence.

### Q2. Would AVA training improve RGNet and A-LAMP?

Likely worth testing, but not verifiable without training.

Local evidence confirms AVA data and splits exist and are much larger than AADB: `data/processed/ava/train_cleaned.csv` has 204,402 data rows, `data/processed/ava/val_cleaned.csv` has 25,551 data rows, and `/home/omen_pc1/ava` contains 255,508 local image files. Local evidence also confirms current RGNet and A-LAMP train scripts can read arbitrary CSV target columns: `/home/omen_pc1/photo_score_project/src/train/train_rgnet.py:27`, `:29`; `/home/omen_pc1/photo_score_project/src/train/train_alamp.py:27`, `:29`.

However, no local RGNet-on-AVA or A-LAMP-on-AVA training result, config, metric file, or exported TFLite artifact was found. Also, AVA manifests expose `mean_score` on a 1..10 scale, while current RGNet and A-LAMP outputs are sigmoid scalar scores in 0..1. Evidence: `data/processed/ava/train_cleaned.csv` header includes `mean_score`; `/home/omen_pc1/photo_score_project/exports/tflite/rgnet_aadb_gpu.metadata.json:15`; `/home/omen_pc1/photo_score_project/exports/tflite/alamp_aadb_gpu.metadata.json:33`. Directly training with `--target_col mean_score` would create a score-scale mismatch unless a normalized AVA scalar manifest or model/postprocessing change is introduced.

### Q3. Can MUSIQ be implemented with current materials?

Paper-inspired implementation possible; prototype only for local WSL use; deployment not recommended.

There is enough current code to train and run a MUSIQ-like model: `/home/omen_pc1/photo_score_project/src/models/musiq.py:1`, `:31`, `:55`; `/home/omen_pc1/photo_score_project/src/train/train_musiq.py:33`; `/home/omen_pc1/photo_score_project/src/datasets/native_size_dataset.py:184`; `/home/omen_pc1/photo_score_project/src/infer/predict_musiq.py:58`. A local checkpoint exists under `/home/omen_pc1/photo_score_project/checkpoints/musiq_aadb_gpu/`.

There is not enough local evidence to reproduce the MUSIQ paper exactly. The implementation itself documents approximations, including learned position/scale embeddings rather than the exact hash-based embedding details: `/home/omen_pc1/photo_score_project/src/models/musiq.py:19`. No MUSIQ TFLite export preset, verification script, metadata JSON, TFLite file, or Flutter usage was found. Evidence: `/home/omen_pc1/photo_score_project/src/export/tflite_presets.py:53` through `:139` lists presets without MUSIQ; `/home/omen_pc1/photo_score_project/src/export/export_tflite.py:14` restricts export to preset choices; `find exports -iname '*musiq*'` returned no files.

### Q4. Which model/version is most suitable?

Primary recommendation: use the existing Stage 5 aesthetic student model as the next mobile aesthetic model, after Flutter integration and target-device smoke/latency testing, and keep KonIQ + FLIVE for technical quality unchanged.

Evidence for the Stage 5 student: `/home/omen_pc1/photo_score_project/docs/pozy_app_stage5_aesthetic_student_handoff.md:5` identifies the checkpoint and TFLite artifact, `:9` reports a 3,411,060-byte built-in-only TFLite file with no SELECT_TF_OPS, `:17` through `:29` define the 224 RGB float32 input and 0..1 scalar output contract, and `:40` through `:48` report local candidate comparison metrics. The same handoff recommends it as the primary app aesthetic candidate at `:51` through `:58`.

Backup recommendation: if the app must stay on the current ensemble path, keep NIMA + RGNet + A-LAMP but fix the A-LAMP preprocessing mismatch first, then re-run score, ranking, TFLite parity, and mobile latency checks. Current Flutter ensemble usage is confirmed in `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/lib/feature/a_cut/layer/inference/aesthetic_model_contract.dart:513`. The A-LAMP WSL preprocessing uses saliency/adaptive patch selection and `resize_with_pad` style global handling, while Flutter currently uses fixed square resize and fixed anchors: `/home/omen_pc1/photo_score_project/src/datasets/native_size_dataset.py:43`, `:60`, `:84`, `:156`; `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/lib/feature/a_cut/layer/inference/image_preprocessor.dart:84`, `:114`.

## 3. NIMA Paper Equivalence Review

| Criterion | Local evidence | Judgment |
|---|---|---|
| 10 score bucket output | Dense 10 softmax head in `/home/omen_pc1/photo_score_project/src/models/nima_distribution.py:37`; TFLite output `[1,10]` in `/home/omen_pc1/photo_score_project/exports/tflite/nima_mobile.verify.json:30` | Matches core NIMA idea |
| Softmax distribution | `/home/omen_pc1/photo_score_project/src/models/nima_distribution.py:37` | Matches |
| EMD or CDF-based loss | `emd_loss` uses `tf.cumsum` in `/home/omen_pc1/photo_score_project/src/models/nima_distribution.py:43`; compile uses it in `/home/omen_pc1/photo_score_project/src/train/train_nima.py:78` | Matches |
| AVA/TID2013/LIVE rating distribution data | AVA distribution dataset reads vote/dist columns in `/home/omen_pc1/photo_score_project/src/datasets/ava_distribution_dataset.py:65` and normalizes labels at `:88` | AVA matches; TID2013/LIVE not verifiable from local evidence for this model |
| ImageNet-pretrained CNN backbone | EfficientNetV2B0 with `weights="imagenet"` in `/home/omen_pc1/photo_score_project/src/models/nima_distribution.py:27` | Matches broad criterion, differs from exact paper reproduction if original backbone/protocol is required |
| Final score as expectation over 1..10 bins | `/home/omen_pc1/photo_score_project/src/models/nima_distribution.py:53`; Flutter expectation and normalization in `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/lib/feature/a_cut/layer/inference/aesthetic_model_contract.dart:339` | Matches |
| Image size/preprocessing | Dataset RGB float [0,1] and train/eval resize behavior in `/home/omen_pc1/photo_score_project/src/datasets/ava_distribution_dataset.py:45`; metadata says 224 RGB float32 `/255` in `/home/omen_pc1/photo_score_project/exports/tflite/nima_mobile.metadata.json:2` | Locally consistent enough for deployment contract; exact paper preprocessing not verifiable |
| TFLite output and postprocessing | `/home/omen_pc1/photo_score_project/exports/tflite/nima_mobile.verify.json:40` reports distribution sum 1.0 and mean score; Flutter contract in `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/lib/feature/a_cut/layer/inference/aesthetic_model_contract.dart:405` | Matches current app contract |

Differences from exact paper reproduction:

- The current file calls the implementation "paper-faithful" for distribution/loss/mean but explicitly says the backbone is approximated with EfficientNetV2B0: `/home/omen_pc1/photo_score_project/src/models/nima_distribution.py:1`, `:19`.
- Local evidence does not prove the exact original training schedule, split policy, optimizer settings, or original paper preprocessing.
- Local evidence only verifies AVA distribution training for this model. TID2013/LIVE training for this specific artifact is not verifiable from local evidence.

Final classification: close implementation.

## 4. RGNet Paper Equivalence Review

| Criterion | Local evidence | Judgment |
|---|---|---|
| DenseNet-121 or FCN feature encoder | Current code uses EfficientNetV2B0 in `/home/omen_pc1/photo_score_project/src/models/rgnet.py:32`; doc says this is an approximation at `:19` | Does not match exact paper criterion |
| Local region feature map | Region projection and node reshape in `/home/omen_pc1/photo_score_project/src/models/rgnet.py:126` through `:132` | Present |
| DenseASPP or multi-scale context module | Not found in local RGNet code | Not verifiable from local evidence |
| Region composition graph | Feature similarity adjacency and normalized graph in `/home/omen_pc1/photo_score_project/src/models/rgnet.py:55` through `:70` | Present in practical form |
| Feature similarity adjacency matrix | Cosine/L2 normalized similarity matmul in `/home/omen_pc1/photo_score_project/src/models/rgnet.py:62` through `:65` | Present |
| Graph convolution | `/home/omen_pc1/photo_score_project/src/models/rgnet.py:76` through `:105` | Present |
| Region-level score aggregation | Region attention and weighted pooling are built in `/home/omen_pc1/photo_score_project/src/models/rgnet.py:105` and used at `:139` | Present as practical weighted aggregation, not proven identical to paper |
| AVA/AADB training setup | Current training script is generic CSV scalar regression; sequential training uses AADB in `/home/omen_pc1/photo_score_project/src/sequential_train.sh:60` through `:67` | AADB current; RGNet-on-AVA not found |
| Loss and metrics | MSE and MAE in `/home/omen_pc1/photo_score_project/src/train/train_rgnet.py:56` through `:61` | Practical scalar regression |
| TFLite ops proving graph reasoning | Previous audit log lists `BATCH_MATMUL`, `L2_NORMALIZATION`, `MATRIX_DIAG`, `SOFTMAX`, and related ops for `rgnet_aadb_gpu.tflite`: `/home/omen_pc1/photo_score_project/outputs/audit_model_paper_basis_20260506/command_log.txt:305` through `:310` | Supports graph-like deployed computation |

Differences from exact paper reproduction:

- The current encoder is EfficientNetV2B0, not DenseNet-121 or the paper FCN encoder: `/home/omen_pc1/photo_score_project/src/models/rgnet.py:19`, `:32`.
- DenseASPP or exact paper multi-scale context was not found.
- The local implementation uses a dense cosine-similarity graph and lightweight GCN, which the source itself labels practical and approximated: `/home/omen_pc1/photo_score_project/src/models/rgnet.py:1` through `:23`.
- Current training is AADB scalar regression with MSE/MAE, not a locally verified reproduction of the full RGNet paper training setup.

Final classification: practical RGNet-style approximation.

## 5. RGNet/A-LAMP AVA Training Feasibility

### Dataset Availability

| Dataset artifact | Local evidence | Feasibility note |
|---|---|---|
| AVA train split | `data/processed/ava/train_cleaned.csv`, 204,402 data rows, header includes `vote_1..vote_10`, `dist_1..dist_10`, `mean_score` | Available |
| AVA val split | `data/processed/ava/val_cleaned.csv`, 25,551 data rows, same label structure | Available |
| AVA test split | `data/processed/ava/test.csv` exists | Available |
| AVA images | `data/raw/ava -> /home/omen_pc1/ava`; `find /home/omen_pc1/ava ...` counted 255,508 image files | Available |
| AADB train/val | `data/processed/aadb/train.csv` has 7,612 data rows; `data/processed/aadb/val.csv` has 846 data rows | Available current scalar dataset |
| SPAQ | `data/raw/spaq` and `data/processed/spaq` directories exist, but no usable CSV or image evidence was found in the inspected commands | Not verifiable from local evidence |

### Script Compatibility

- `train_rgnet.py` can accept an arbitrary CSV and target column: `/home/omen_pc1/photo_score_project/src/train/train_rgnet.py:27` through `:34`.
- `train_alamp.py` can accept an arbitrary CSV and target column: `/home/omen_pc1/photo_score_project/src/train/train_alamp.py:27` through `:37`.
- The shared CSV dataset loader expects `image_path` and the requested target column: `/home/omen_pc1/photo_score_project/src/datasets/csv_dataset.py:21` through `:37`.
- A-LAMP native-size dataset also expects `image_path` and target column: `/home/omen_pc1/photo_score_project/src/datasets/native_size_dataset.py:244` through `:285`.

Important compatibility limit: existing RGNet and A-LAMP models output sigmoid scalar scores in 0..1. AVA `mean_score` is 1..10. A normalized scalar AVA manifest, such as `(mean_score - 1) / 9`, is required before the current scripts should be used for AVA scalar training. A local normalized AVA scalar manifest for RGNet/A-LAMP was not found.

### Existing AVA Training Evidence

- NIMA uses AVA distribution labels: `/home/omen_pc1/photo_score_project/src/train/train_nima.py:42` through `:63`.
- Current sequential training for RGNet and A-LAMP uses AADB: `/home/omen_pc1/photo_score_project/src/sequential_train.sh:45` through `:52` and `:60` through `:67`.
- No local RGNet-on-AVA or A-LAMP-on-AVA script/config/metrics/export artifact was found.

### Strategy Analysis

| Strategy | Expected benefit | Local evidence | Required changes | Training cost | Deployment risk | Capstone fit |
|---|---|---|---|---|---|---|
| AADB-only current RGNet/A-LAMP | Already integrated and exported | AADB scripts and TFLite files exist | None | None | Existing risks only; A-LAMP preprocessing mismatch remains | Good as current baseline |
| AVA-only RGNet | Likely worth testing, not locally proven | AVA data exists; RGNet script can read CSV target | Normalized AVA scalar column or model output scale change | Medium; exact time not verifiable | Medium; TFLite path exists for RGNet style but retrained export still must be verified | Reasonable if time remains |
| AVA-only A-LAMP | Likely worth testing, not locally proven | AVA data exists; A-LAMP script can read CSV target | Normalized AVA scalar column; preprocessing parity must be fixed | Medium to high; exact time not verifiable | Medium to high due multi-input preprocessing/export | Secondary after preprocessing fix |
| AVA pretrain then AADB fine-tune | Plausible but not locally proven | AVA and AADB exist | Train scripts do not expose a resume/load-weights argument in inspected parser lines; implementation change likely | High | Medium | Only if timeline allows |
| AVA + AADB mixed training | Unknown | Both datasets exist | Unified normalized scale, sampling/balancing, and validation design needed | Medium to high | Medium | Not first-line for deadline |
| Teacher/student distillation | Strong local deployment evidence through Stage 5 student | Stage 5 student artifact and comparison report exist | Flutter integration and target-device smoke test | Low if using existing student; higher if retraining | Low TFLite risk for existing student | Best capstone-safe path |
| Keep current models and only fix A-LAMP preprocessing | Unknown accuracy gain, but fixes a confirmed contract mismatch | WSL adaptive/saliency preprocessing differs from Flutter fixed crop | Flutter preprocessing change later, then comparison test | Low training cost | Low model risk; app risk medium | Best first experiment if staying on ensemble |

### Metrics For Comparison

Use at least:

- SRCC/Spearman: ranking quality for photo selection.
- PLCC/Pearson: linear score agreement.
- MAE and MSE/RMSE: calibration and absolute score error.
- Binary accuracy around a chosen score threshold: only after a threshold is defined locally.
- Top-k agreement for A-cut selection: locally supported by existing evaluation utilities.
- Pairwise ranking agreement: locally supported by existing evaluation utilities.
- TFLite parity: Keras/SavedModel/TFLite score difference and ranking preservation.
- Mobile latency and memory: required on Galaxy S23 Ultra or target-like device; not verifiable from current WSL evidence.

Final recommendation for AVA: run controlled experiments only after normalizing AVA scalar labels and defining evaluation metrics. Do not state that AVA improves RGNet or A-LAMP until local training/evaluation results exist.

## 6. MUSIQ Feasibility Review

### Current Code Status

| MUSIQ paper-related item | Local evidence | Judgment |
|---|---|---|
| Patch-based Transformer | Multi-head attention block in `/home/omen_pc1/photo_score_project/src/models/musiq.py:31`; model builder in `:55` | Present |
| Native/aspect-preserving multi-scale input | Dataset token builder resizes by long side and extracts scale tokens in `/home/omen_pc1/photo_score_project/src/datasets/native_size_dataset.py:184` through `:241` | Present in practical form |
| Multi-scale representation | Default scales `224,384,512` in `/home/omen_pc1/photo_score_project/src/train/train_musiq.py:42` | Present |
| Patch size around 32 | Default `--patch_size 32` in `/home/omen_pc1/photo_score_project/src/train/train_musiq.py:41` | Present |
| Hash-based 2D spatial embedding | Source documents learned position/scale embeddings instead of exact hash-based details in `/home/omen_pc1/photo_score_project/src/models/musiq.py:19` | Exact paper detail not present from local evidence |
| Scale embedding | Scale embedding in `/home/omen_pc1/photo_score_project/src/models/musiq.py:95` | Present |
| Transformer depth/hidden/head count | Defaults: embed 128, depth 4, heads 4, MLP 256 in `/home/omen_pc1/photo_score_project/src/train/train_musiq.py:43` through `:47` | Present but practical size, not exact paper reproduction |
| ImageNet pretraining | No local evidence found in MUSIQ code/checkpoint metadata | Not verifiable from local evidence |
| IQA or aesthetic fine-tuning dataset | Training script accepts CSV target; local AADB/KonIQ/PaQ data exists | Feasible, exact paper setup not proven |
| TFLite export | No MUSIQ preset in `/home/omen_pc1/photo_score_project/src/export/tflite_presets.py:53` through `:139`; export uses preset choices in `/home/omen_pc1/photo_score_project/src/export/export_tflite.py:14` | Not available locally |
| Flutter usage | No MUSIQ asset or active Flutter loading evidence found | Not used |

### Direct MUSIQ Answers

1. Enough code to train a MUSIQ-like model: yes, from local WSL evidence.
2. Enough code to reproduce the MUSIQ paper exactly: no; not verifiable from local evidence.
3. Enough infrastructure to export MUSIQ to TFLite: no; no local MUSIQ export preset, metadata, verify file, or TFLite artifact was found.
4. Suitable for on-device Flutter inference: deployment not recommended from local evidence. TFLite export and latency are unproven, and the model is transformer/multi-input style.
5. Best use: WSL/offline evaluator or server-side teacher candidate. Mobile deployment should wait until TFLite export, parity, and target-device latency are proven.

Final classification: paper-inspired implementation possible; prototype only for deployment purposes.

## 7. Model/Version Decision Matrix

| Option | Expected accuracy impact | Evidence strength | Implementation difficulty | Training cost | TFLite risk | Mobile latency risk | App integration risk | Reportability | Final recommendation |
|---|---|---|---|---|---|---|---|---|---|
| 1. Current NIMA + RGNet + A-LAMP ensemble | Medium, not locally compared against Stage 5 as active ensemble | High for existence/usage | Low | Low | Low to medium | Medium, three aesthetic models and A-LAMP multi-input | Low for current app, medium for quality | Good if described as practical ensemble | Acceptable baseline |
| 2. Current ensemble but fix A-LAMP preprocessing mismatch | Unknown to medium; fixes confirmed mismatch | Medium | Medium | Low | Low | Medium | Medium | Strong because it fixes verified mismatch | Backup recommendation |
| 3. Retrain RGNet and A-LAMP on AVA | Unknown; likely worth testing | Low until experiments run | Medium | Medium to high | Medium | Medium | Medium | Reportable only with controlled metrics | Experiment only |
| 4. AVA pretrain then AADB fine-tune | Unknown to medium-high | Low | High | High | Medium | Medium | Medium to high | Good if completed, risky for deadline | Later experiment |
| 5. Add MUSIQ as fourth aesthetic model | Unknown | Low | High | High | High | High | High | Risky without TFLite/latency | Not recommended now |
| 6. Replace ensemble with MUSIQ | Unknown | Low | High | High | High | High | High | Unsafe claim locally | Not recommended now |
| 7. Use MUSIQ only as teacher/offline evaluator | Medium potential, not proven | Medium for code/checkpoint existence | Medium | Medium | None for app if offline only | None for app if offline only | Low | Capstone-safe if labeled offline/teacher | Optional later |
| 8. Use existing Stage 5 smaller student model | High deployment fit for existing Stage 5 student | Medium to high | Low to medium | Low if using existing Stage 5 | Low for existing Stage 5 TFLite | Low | Medium until Flutter integration | Strong if claims cite local handoff metrics | Primary recommendation |
| 9. Keep KonIQ + FLIVE technical quality unchanged | Stable, avoids unrelated risk | High for current artifacts/contracts | Low | Low | Low | Low to medium | Low | Safe as dataset-based technical scoring | Recommended |
| 10. Replace or retrain technical models | Unknown | Low | High | High | Medium to high | Unknown | Medium to high | Not justified by local evidence | Not recommended now |

Primary recommendation: Stage 5 student aesthetic model plus unchanged KonIQ + FLIVE technical quality, after Flutter integration and target-device smoke/latency testing.

Backup recommendation: current NIMA + RGNet + A-LAMP ensemble, but fix A-LAMP preprocessing mismatch before treating ensemble output as stable.

Not recommended now: mobile MUSIQ deployment, replacing the ensemble with MUSIQ, or retraining technical quality models without a new local benchmark.

## 8. Recommended Experiment Plan

Detailed experiment commands and rollback criteria are in `outputs/model_strategy_review_20260506/experiment_plan.md`.

Recommended order:

1. Fix A-LAMP preprocessing mismatch and compare before/after.
2. Train/evaluate RGNet on AVA only if script compatibility is confirmed with normalized 0..1 scalar labels.
3. Train/evaluate A-LAMP on AVA only if script compatibility is confirmed with normalized 0..1 scalar labels.
4. Test AVA pretrain + AADB fine-tune only after the AVA-only runs produce useful metrics.
5. Only then consider MUSIQ TFLite export or MUSIQ teacher/offline use.

## 9. Final Safe Capstone Wording

다음 문구는 로컬 파일과 산출물로 확인 가능한 범위에 맞춘 보수적 표현이다.

> 본 프로젝트의 미학 평가 파이프라인은 NIMA, RGNet-style, A-LAMP-style 모델을 활용한 실용적 앙상블 구조를 사용하였다. NIMA 모델은 10개 점수 분포, softmax 출력, EMD/CDF 기반 손실, AVA 평점 분포 학습, 기대값 기반 점수 산출 구조를 포함하므로 NIMA 논문의 핵심 아이디어에 가까운 구현으로 볼 수 있다. 다만 현재 백본과 학습 세부 조건이 원 논문과 완전히 동일하다는 근거는 로컬 산출물만으로 확인되지 않으므로, 완전한 논문 재현이라고 표현하지 않는다.

> RGNet-style 모델은 지역 특징, 특징 유사도 기반 인접 행렬, graph convolution, 지역 가중 집계를 포함하여 RGNet의 핵심 개념을 실용적으로 반영하였다. 그러나 현재 구현은 원 논문에서 제시된 DenseNet-121 또는 FCN/DenseASPP 기반 구조가 아니라 EfficientNetV2B0 기반의 경량화된 그래프 구조이므로, 정확한 RGNet 논문 재현이 아니라 모바일 적용을 고려한 실용적 근사 구현으로 기술한다.

> A-LAMP-style 모델은 전역 이미지 입력과 다중 패치 입력, 패치 가중 집계, 레이아웃 cue를 포함하지만, 원 논문의 adaptive patch selection 및 layout-aware attribute graph 전체 구조를 동일하게 구현했다는 근거는 부족하다. 따라서 직접 구현이 아니라 논문 아이디어를 반영한 응용 구현으로 기술한다.

> AVA 데이터셋을 이용한 RGNet/A-LAMP 재학습은 데이터와 스크립트 관점에서 실험 가능성이 있으나, 성능 향상은 로컬 실험 결과 없이는 단정할 수 없다. 향상 여부는 동일한 검증셋에서 SRCC, PLCC, MAE, top-k agreement, TFLite parity, 모바일 지연시간을 비교하여 판단해야 한다.

> MUSIQ는 WSL 프로젝트에 학습 가능한 prototype 코드와 checkpoint가 존재하지만, 현재 Flutter 앱에서 사용되지 않으며 TFLite 변환과 모바일 지연시간이 검증되지 않았다. 따라서 현 단계에서는 모바일 배포 모델보다는 오프라인 평가기 또는 teacher 모델 후보로 제한하여 다루는 것이 안전하다.

> 최종 모바일 적용 관점에서는 기존 TFLite 기반 파이프라인을 유지하되, 미학 점수는 검증된 경량 Stage 5 student 모델 또는 현재 앙상블의 전처리 정합성 개선 버전을 우선 검토하는 것이 적절하다. 기술 품질 평가는 현재 KonIQ 및 FLIVE 기반 TFLite 모델을 유지하는 것이 구현 위험과 일정 측면에서 가장 보수적인 선택이다.
