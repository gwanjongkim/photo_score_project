# 감사 파일 인벤토리 (Full Audit Index)

보고서 작성을 위해 확인한 주요 파일과 그 의미, 그리고 보고서 반영 여부를 기록합니다.

| 파일 경로 | 파일의 의미 | 보고서 반영 여부 |
| :--- | :--- | :--- |
| `outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv` | 모든 평가된 모델의 성능(SRCC, Accuracy 등) 지표 총괄 요약본 | 반영 (주요 성능, 모델 상태표 작성) |
| `outputs/acut_weekly_research_journal_v2_20260526/weekly_research_journal_combined.md` | 이전 연구진이 작성한 주차별 로우 데이터 및 수행 작업 목록 | 반영 (타임라인 재구성 및 실험 흐름 파악) |
| `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/report.md` | TOPIQ-lite와 FLIVE, KonIQ 비교 벤치마크 결과 보고서 | 반영 (8주차 기술 품질 모델 성능 근거) |
| `outputs/distill_conservative_aesthetic_full_aadb_20260423/...` | Stage 5 Student 모델 증류 학습 및 평가 결과 | 반영 (4주차 실험 내용, 단 **최종 사용 안 함** 명시) |
| `outputs/gemini_api_teacher_aesthetic3_20260427_revised/...` | Gemma 어댑터 QLoRA 학습 결과 및 환각(Hallucination) 이슈 기록 | 반영 (5주차 실패/보류 근거) |
| `outputs/alamp_paper_fusion_4096_final_report.md` | A-LAMP 4096 서브셋 대상 그래프 퓨전 시도 및 과적합(Overfit) 실패 기록 | 반영 (7주차 실패 근거) |
| `outputs/rgnet_v1_cascaded_denseaspp.../report.md` | RGNet DenseASPP 추가 실험 및 베이스라인 미달 기록 | 반영 (6주차/8주차 실패 근거) |
| `outputs/eval_techiqa_guard_v1...` | DistortionGuard 데이터 편향 이슈 및 FLIVE 오탐지 분석 | 반영 (8주차 실패 근거) |
| `outputs/icaa_performance_reverification_20260525/report.md` | ICAA Color 미학 모델의 SRCC 성능 및 TF Native 변환 검증 | 반영 (9주차 성능 증명) |
| `outputs/nima_fixed_preproc_eval_20260525/report.md` | NIMA EfficientNetV2 이중 스케일링 전처리 버그 수정 리포트 | 반영 (9주차 성능 향상 근거) |

총 500여 개 이상의 개별 `summary.json`, `predictions.csv`, `top_k_summary.json` 파일을 메타데이터로 참조하여 전체 실험 타임라인을 교차 검증하였습니다.
