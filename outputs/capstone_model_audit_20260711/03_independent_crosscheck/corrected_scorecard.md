<!-- 모델별 재현성 및 실험 타당성을 독립적으로 재평가한 보정 스코어카드를 제공하는 문서. -->
# Corrected Scorecard

This scorecard corrects the paper fidelity and experimental validity scores of the models based on the adversarial review findings.

## Scorecard Table

Scores are graded out of 100. A score of 0 denotes that the model is completely unimplemented or has no codebase footprint.

| Model | Codex Paper Fidelity | Corrected Paper Fidelity | Codex Exp Validity | Corrected Exp Validity | Production Suitability | Traceability | Confidence | Adversarial Classification |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **KonIQ mobile** | 15 | **5** | 58 | **45** | 72 | 63 | HIGH | PRODUCTION_ADAPTED_MODEL |
| **FLIVE image mobile** | 10 | **5** | 50 | **40** | 70 | 61 | HIGH | PRODUCTION_ADAPTED_MODEL |
| **FLIVE patch** | 8 | **5** | 46 | **38** | 52 | 55 | MEDIUM | PAPER_INSPIRED_IMPLEMENTATION |
| **Technical ensemble** | 5 | **5** | 48 | **40** | 60 | 44 | HIGH | PRODUCTION_ADAPTED_MODEL |
| **NIMA fixed candidate** | 72 | **68** | 64 | **64** | 70 | 67 | HIGH | SUBSTANTIALLY_FAITHFUL |
| **NIMA legacy/deployed** | 58 | **50** | 35 | **30** | 38 | 62 | HIGH | EXPERIMENTAL_PROTOTYPE |
| **RGNet practical** | 48 | **18** | 63 | **45** | 66 | 65 | HIGH | EXPERIMENTAL_PROTOTYPE |
| **RGNet paper-v1/PIL** | 65 | **50** | 58 | **50** | 48 | 54 | MEDIUM | PAPER_INSPIRED_IMPLEMENTATION |
| **A-LAMP practical** | 40 | **30** | 57 | **50** | 58 | 59 | HIGH | PAPER_INSPIRED_IMPLEMENTATION |
| **Mobile A-LAMP v2** | 36 | **30** | 68 | **68** | 82 | 78 | HIGH | PRODUCTION_ADAPTED_MODEL |
| **Aesthetic ensemble** | 0 | **0** | 62 | **45** | 45 | 48 | MEDIUM | PRODUCTION_ADAPTED_MODEL |
| **ICAA/DAT local line** | 25 | **0** | 35 | **0** | 30 | 37 | LOW | NOT_IMPLEMENTED |
| **MUSIQ practical AADB** | 38 | **30** | 40 | **35** | 42 | 59 | HIGH | PAPER_INSPIRED_IMPLEMENTATION |
| **MUSIQ technical probes** | 25 | **15** | 30 | **25** | 25 | 43 | HIGH | EXPERIMENTAL_PROTOTYPE |
| **MDIQA teacher** | 78 | **78** | 50 | **50** | 20 | 55 | MEDIUM | SUBSTANTIALLY_FAITHFUL |
| **MDIQA 14k/50k students** | 10 | **5** | 42 | **35** | 30 | 58 | HIGH | EXPERIMENTAL_PROTOTYPE |
| **Stage 5 aesthetic student** | 0 | **0** | 35 | **30** | 45 | 53 | HIGH | EXPERIMENTAL_PROTOTYPE |
| **TOPIQ-lite/C6/RankCalib** | 35 | **25** | 52 | **45** | 45 | 56 | MEDIUM | PAPER_INSPIRED_IMPLEMENTATION |
| **TechIQA-Guard** | 0 | **0** | 45 | **40** | 10 | 62 | HIGH | EXPERIMENTAL_PROTOTYPE |
| **DistortionGuard** | 0 | **0** | 52 | **45** | 10 | 64 | HIGH | EXPERIMENTAL_PROTOTYPE |

## Scoring Rationale

1. **ICAA/DAT local line**: Downgraded to **0** because the local module `src/models/icaa_dat.py` does not exist. It is completely unimplemented locally.
2. **RGNet practical**: Downgraded paper fidelity to **18** and experimental validity to **45**. Dilated convolutions, DenseASPP context, learned projection matrices, and LSE aggregation are all missing. The code also contains the double-scaling bug (internal preprocessing enabled while receiving `[0, 1]` inputs), compressing inputs to `[0, 0.0039]`.
3. **KonIQ / FLIVE image mobile**: Downgraded paper fidelity to **5** because they are simple MobileNetV2 regressors trained on random splits, swapping the entire paper architectures (Inception-ResNet-v2 and PaQ-2-PiQ ROI). Experimental validity is downgraded due to MobileNetV2 ImageNet scale mismatch (`[0, 1]` input vs `[-1, 1]` pre-trained expectation).
4. **NIMA fixed candidate**: Downgraded slightly to **68** due to the pre-trained ImageNet backbone being trained directly on `[0, 1]` inputs without calling `preprocess_input`.
5. **FLIVE split leakage**: Downgraded experimental validity of all FLIVE models (including students) due to image/patch overlap leakage inside `make_flive_csv.py`.
