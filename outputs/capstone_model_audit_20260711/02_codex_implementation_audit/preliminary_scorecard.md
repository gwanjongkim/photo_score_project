# Preliminary scorecard

Scores reflect current evidence, not model quality alone.

| Model | Paper fidelity | Experimental validity | Production suitability | Traceability | Confidence | Classification |
|---|---:|---:|---:|---:|---|---|
| KonIQ mobile | 15 | 58 | 72 | 63 | HIGH | PRODUCTION_ADAPTED_MODEL |
| FLIVE image mobile | 10 | 50 | 70 | 61 | HIGH | PRODUCTION_ADAPTED_MODEL |
| FLIVE patch | 8 | 46 | 52 | 55 | MEDIUM | PAPER_INSPIRED_IMPLEMENTATION |
| Technical ensemble | 5 | 48 | 60 | 44 | HIGH | PRODUCTION_ADAPTED_MODEL |
| NIMA fixed candidate | 72 | 64 | 70 | 67 | HIGH | SUBSTANTIALLY_FAITHFUL |
| NIMA legacy/deployed | 58 | 35 | 38 | 62 | HIGH | EXPERIMENTAL_PROTOTYPE |
| RGNet practical | 48 | 63 | 66 | 65 | HIGH | PAPER_INSPIRED_IMPLEMENTATION |
| RGNet paper-v1/PIL candidate family | 65 | 58 | 48 | 54 | MEDIUM | PAPER_INSPIRED_IMPLEMENTATION |
| A-LAMP practical | 40 | 57 | 58 | 59 | HIGH | PAPER_INSPIRED_IMPLEMENTATION |
| Mobile A-LAMP v2 | 36 | 68 | 82 | 78 | HIGH | PRODUCTION_ADAPTED_MODEL |
| Aesthetic ensemble | 0 | 62 | 45 | 48 | MEDIUM | PRODUCTION_ADAPTED_MODEL |
| ICAA/DAT local line | 25 | 35 | 30 | 37 | LOW | UNVERIFIED |
| MUSIQ practical AADB | 38 | 40 | 42 | 59 | HIGH | PAPER_INSPIRED_IMPLEMENTATION |
| MUSIQ technical probes | 25 | 30 | 25 | 43 | HIGH | EXPERIMENTAL_PROTOTYPE |
| MDIQA teacher | 78 | 50 | 20 | 55 | MEDIUM | SUBSTANTIALLY_FAITHFUL |
| MDIQA 14k/50k students | 10 | 42 | 30 | 58 | HIGH | EXPERIMENTAL_PROTOTYPE |
| Stage 5 aesthetic student | 0 | 35 | 45 | 53 | HIGH | EXPERIMENTAL_PROTOTYPE |
| TOPIQ-lite/C6/RankCalib | 35 | 52 | 45 | 56 | MEDIUM | PAPER_INSPIRED_IMPLEMENTATION |
| TechIQA-Guard | 0 | 45 | 10 | 62 | HIGH | EXPERIMENTAL_PROTOTYPE |
| DistortionGuard | 0 | 52 | 10 | 64 | HIGH | EXPERIMENTAL_PROTOTYPE |

No model receives `PAPER_FAITHFUL_REPRODUCTION`: none has complete paper architecture, official protocol, checkpoint/export/runtime lineage and independently recomputable metrics simultaneously.
