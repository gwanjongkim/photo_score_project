# 1주차 연구일지: 2026-03-30 ~ 2026-04-05

## 1. 주차 목표

Stage 5 및 Firebase 초기 통합

## 2. 수행 작업

### Aesthetic Models
- `configs/stage5_reference.json` 작업 확인
- `outputs/stage5_comparisons/stage5_compare_smoke/comparison_summary.csv` 작업 확인
- `outputs/stage5_comparisons/stage5_compare_smoke/comparison_summary.json` 작업 확인
- `outputs/stage5_runs/stage5_smoke_test/ranking_summary.json` 작업 확인
- `outputs/stage5_runs/stage5_smoke_test/run_manifest.json` 작업 확인
- `docs/acut_product_integration.md` 작업 확인
- `outputs/acut_product_contract_check/top_k_summary.json` 작업 확인
- `outputs/acut_product_contract_check_diversity/top_k_summary.json` 작업 확인
- `outputs/acut_product_contract_check_smoke_diversity/top_k_summary.json` 작업 확인
- `outputs/acut_product_firebase_regression/top_k_summary.json` 작업 확인
- 외 3개 파일 변동

### Technical Quality
- `configs/stage5_reference.json` 작업 확인
- `outputs/stage5_runs/stage5_smoke_test/run_manifest.json` 작업 확인

### VLM/Reasoning
- `docs/acut_product_integration.md` 작업 확인
- `outputs/acut_product_contract_check/top_k_summary.json` 작업 확인
- `outputs/acut_product_contract_check_diversity/top_k_summary.json` 작업 확인
- `outputs/acut_product_contract_check_smoke_diversity/top_k_summary.json` 작업 확인
- `outputs/acut_product_firebase_regression/top_k_summary.json` 작업 확인
- `outputs/firebase_worker_local_smoke/top_k_summary.json` 작업 확인
- `outputs/firebase_worker_local_smoke_diversity/top_k_summary.json` 작업 확인
- `outputs/acut_product_ready_smoke_diversity_post_integration/top_k_summary.json` 작업 확인
- `outputs/acut_product_ready_smoke_post_integration/top_k_summary.json` 작업 확인
- `outputs/acut_verify_now/top_k_summary.json` 작업 확인
- 외 1개 파일 변동

### Mobile/Deployment
- `docs/acut_product_integration.md` 작업 확인

### Firebase/Backend
- `outputs/firebase_worker_local_smoke/top_k_summary.json` 작업 확인
- `outputs/firebase_worker_local_smoke_diversity/top_k_summary.json` 작업 확인
- `outputs/firebase_worker_local_smoke_post_integration/top_k_summary.json` 작업 확인

## 3. 실험 및 구현 내용

Stage 5 모델의 초기 스모크 테스트와 Firebase Worker 연동 실험이 수행됨.

## 4. 결과

- Firebase Worker 로컬 스모크 테스트 성공 (EVE_0010)
- Acut 제품 계약 체크 (top_k_summary.json 생성, EVE_0006)

## 5. 실패 / 문제 / 보류 사항

- 근거 파일 미확인

## 6. 의사결정

- Firebase를 작업 큐로 사용하고 Python 워커를 별도 분리하기로 결정.

## 7. 산출물

- `configs/stage5_reference.json`
- `docs/acut_product_integration.md`
- `outputs/acut_product_contract_check/top_k_summary.json`
- `outputs/acut_product_contract_check_diversity/top_k_summary.json`
- `outputs/acut_product_contract_check_smoke_diversity/top_k_summary.json`
- `outputs/acut_product_firebase_regression/top_k_summary.json`
- `outputs/acut_product_ready_smoke_diversity_post_integration/top_k_summary.json`
- `outputs/acut_product_ready_smoke_post_integration/top_k_summary.json`
- `outputs/acut_verify_now/top_k_summary.json`
- `outputs/firebase_worker_local_smoke/top_k_summary.json`
- `outputs/firebase_worker_local_smoke_diversity/top_k_summary.json`
- `outputs/firebase_worker_local_smoke_post_integration/top_k_summary.json`
- `outputs/stage5_comparisons/stage5_compare_smoke/comparison_summary.csv`
- `outputs/stage5_comparisons/stage5_compare_smoke/comparison_summary.json`
- `outputs/stage5_runs/stage5_smoke_test/ranking_summary.json`

## 8. 다음 주 계획

Acut 제품 백엔드 정비 및 증류 학습

## 9. 근거 목록

- [EVE_0000] configs/stage5_reference.json (날짜: 2026-04-02, 소스: content)
- [EVE_0001] outputs/stage5_comparisons/stage5_compare_smoke/comparison_summary.csv (날짜: 2026-04-02, 소스: content)
- [EVE_0002] outputs/stage5_comparisons/stage5_compare_smoke/comparison_summary.json (날짜: 2026-04-02, 소스: content)
- [EVE_0003] outputs/stage5_runs/stage5_smoke_test/ranking_summary.json (날짜: 2026-04-02, 소스: content)
- [EVE_0004] outputs/stage5_runs/stage5_smoke_test/run_manifest.json (날짜: 2026-04-02, 소스: content)
- [EVE_0005] docs/acut_product_integration.md (날짜: 2026-04-03, 소스: content)
- [EVE_0006] outputs/acut_product_contract_check/top_k_summary.json (날짜: 2026-04-03, 소스: content)
- [EVE_0007] outputs/acut_product_contract_check_diversity/top_k_summary.json (날짜: 2026-04-03, 소스: content)
- [EVE_0008] outputs/acut_product_contract_check_smoke_diversity/top_k_summary.json (날짜: 2026-04-03, 소스: content)
- [EVE_0009] outputs/acut_product_firebase_regression/top_k_summary.json (날짜: 2026-04-03, 소스: content)
- [EVE_0010] outputs/firebase_worker_local_smoke/top_k_summary.json (날짜: 2026-04-03, 소스: content)
- [EVE_0011] outputs/firebase_worker_local_smoke_diversity/top_k_summary.json (날짜: 2026-04-03, 소스: content)
- [EVE_0012] outputs/acut_product_ready_smoke_diversity_post_integration/top_k_summary.json (날짜: 2026-04-04, 소스: content)
- [EVE_0013] outputs/acut_product_ready_smoke_post_integration/top_k_summary.json (날짜: 2026-04-04, 소스: content)
- [EVE_0014] outputs/acut_verify_now/top_k_summary.json (날짜: 2026-04-04, 소스: content)
- [EVE_0015] outputs/firebase_worker_local_smoke_post_integration/top_k_summary.json (날짜: 2026-04-04, 소스: content)
# 2주차 연구일지: 2026-04-06 ~ 2026-04-12

## 1. 주차 목표

Acut 제품 백엔드 정비 및 증류 학습

## 2. 수행 작업

### VLM/Reasoning
- `outputs/acut_product_backend_cleanup_smoke/top_k_summary.json` 작업 확인
- `outputs/acut_product_gemini_backend_sample/top_k_summary.json` 작업 확인
- `outputs/acut_product_ready_smoke_contract/top_k_summary.json` 작업 확인
- `outputs/acut_product_ready_smoke_contract_wrapper_skip/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_probe/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_format_check/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_format_check_v2/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_format_check_v3/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_format_check_v4/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_format_check_v5/top_k_summary.json` 작업 확인
- 외 8개 파일 변동

### Aesthetic Models
- `outputs/acut_product_backend_cleanup_smoke/top_k_summary.json` 작업 확인
- `outputs/acut_product_gemini_backend_sample/top_k_summary.json` 작업 확인
- `outputs/acut_product_ready_smoke_contract/top_k_summary.json` 작업 확인
- `outputs/acut_product_ready_smoke_contract_wrapper_skip/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_probe/top_k_summary.json` 작업 확인
- `outputs/distill_exec_20260411/aadb_subset_teacher_labels.csv` 작업 확인
- `outputs/distill_exec_20260411/stage5_student_aadb_subset/evaluation/test_predictions.csv` 작업 확인
- `outputs/distill_exec_20260411/stage5_student_aadb_subset/evaluation/val_predictions.csv` 작업 확인
- `outputs/distill_exec_20260411/stage5_student_aadb_subset/manual_evaluation/summary.json` 작업 확인
- `outputs/distill_exec_20260411/stage5_student_aadb_subset/manual_evaluation/test_predictions.csv` 작업 확인
- 외 16개 파일 변동

### Mobile/Deployment
- `outputs/distill_exec_20260411/stage5_student_aadb_subset/student_model.float16.metadata.json` 작업 확인
- `outputs/distill_exec_20260411/stage5_student_aadb_subset/tflite_smoke_comparison.json` 작업 확인
- `outputs/distill_gpu_runtime_check/runtime_info.txt` 작업 확인

## 3. 실험 및 구현 내용

백엔드 정리 작업 및 Stage 5 학생 모델 증류 학습 실행 (distill_exec).

## 4. 결과

- AADB 서브셋 대상 교사 레이블 생성 (EVE_2511)
- Stage 5 학생 모델 테스트 예측 수행 (EVE_0525)

## 5. 실패 / 문제 / 보류 사항

- 근거 파일 미확인

## 6. 의사결정

- 데이터 기반 성능 고도화 지속.

## 7. 산출물

- `outputs/acut_product_backend_cleanup_smoke/top_k_summary.json`
- `outputs/acut_product_gemini_backend_sample/top_k_summary.json`
- `outputs/acut_product_ready_smoke_contract/top_k_summary.json`
- `outputs/acut_product_ready_smoke_contract_wrapper_skip/top_k_summary.json`
- `outputs/acut_product_vila_local_format_check/top_k_summary.json`
- `outputs/acut_product_vila_local_format_check_v2/top_k_summary.json`
- `outputs/acut_product_vila_local_format_check_v3/top_k_summary.json`
- `outputs/acut_product_vila_local_format_check_v4/top_k_summary.json`
- `outputs/acut_product_vila_local_format_check_v5/top_k_summary.json`
- `outputs/acut_product_vila_local_format_check_v6/top_k_summary.json`
- `outputs/acut_product_vila_local_format_check_v7/top_k_summary.json`
- `outputs/acut_product_vila_local_probe/top_k_summary.json`
- `outputs/acut_product_vila_local_real/top_k_summary.json`
- `outputs/acut_product_vila_local_real2/top_k_summary.json`
- `outputs/acut_product_vila_local_score_reasoning_v1/top_k_summary.json`

## 8. 다음 주 계획

VILA 및 NVILA-8B 멀티모달 추론 실험

## 9. 근거 목록

- [EVE_0016] outputs/acut_product_backend_cleanup_smoke/top_k_summary.json (날짜: 2026-04-11, 소스: content)
- [EVE_0017] outputs/acut_product_gemini_backend_sample/top_k_summary.json (날짜: 2026-04-11, 소스: content)
- [EVE_0018] outputs/acut_product_ready_smoke_contract/top_k_summary.json (날짜: 2026-04-11, 소스: content)
- [EVE_0019] outputs/acut_product_ready_smoke_contract_wrapper_skip/top_k_summary.json (날짜: 2026-04-11, 소스: content)
- [EVE_0020] outputs/acut_product_vila_local_probe/top_k_summary.json (날짜: 2026-04-11, 소스: content)
- [EVE_0021] outputs/distill_exec_20260411/aadb_subset_teacher_labels.csv (날짜: 2026-04-11, 소스: filename)
- [EVE_0022] outputs/distill_exec_20260411/stage5_student_aadb_subset/evaluation/test_predictions.csv (날짜: 2026-04-11, 소스: filename)
- [EVE_0023] outputs/distill_exec_20260411/stage5_student_aadb_subset/evaluation/val_predictions.csv (날짜: 2026-04-11, 소스: filename)
- [EVE_0024] outputs/distill_exec_20260411/stage5_student_aadb_subset/manual_evaluation/summary.json (날짜: 2026-04-11, 소스: filename)
- [EVE_0025] outputs/distill_exec_20260411/stage5_student_aadb_subset/manual_evaluation/test_predictions.csv (날짜: 2026-04-11, 소스: filename)
- [EVE_0026] outputs/distill_exec_20260411/stage5_student_aadb_subset/manual_evaluation/val_predictions.csv (날짜: 2026-04-11, 소스: filename)
- [EVE_0027] outputs/distill_exec_20260411/stage5_student_aadb_subset/resolved_config.json (날짜: 2026-04-11, 소스: filename)
- [EVE_0028] outputs/distill_exec_20260411/stage5_student_aadb_subset/student_model.float16.metadata.json (날짜: 2026-04-11, 소스: filename)
- [EVE_0029] outputs/distill_exec_20260411/stage5_student_aadb_subset/tflite_smoke_comparison.json (날짜: 2026-04-11, 소스: filename)
- [EVE_0030] outputs/acut_product_vila_local_format_check/top_k_summary.json (날짜: 2026-04-12, 소스: content)
- [EVE_0031] outputs/acut_product_vila_local_format_check_v2/top_k_summary.json (날짜: 2026-04-12, 소스: content)
- [EVE_0032] outputs/acut_product_vila_local_format_check_v3/top_k_summary.json (날짜: 2026-04-12, 소스: content)
- [EVE_0033] outputs/acut_product_vila_local_format_check_v4/top_k_summary.json (날짜: 2026-04-12, 소스: content)
- [EVE_0034] outputs/acut_product_vila_local_format_check_v5/top_k_summary.json (날짜: 2026-04-12, 소스: content)
- [EVE_0035] outputs/acut_product_vila_local_format_check_v6/top_k_summary.json (날짜: 2026-04-12, 소스: content)
- 외 8개 근거 파일 존재
# 3주차 연구일지: 2026-04-13 ~ 2026-04-19

## 1. 주차 목표

VILA 및 NVILA-8B 멀티모달 추론 실험

## 2. 수행 작업

### VLM/Reasoning
- `outputs/acut_product_nvila8b_probe_v3/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v10/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v5/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v6/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v7/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v8/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v9/top_k_summary.json` 작업 확인
- `outputs/acut_product_nvila8b_probe_v10_gpu/top_k_summary.json` 작업 확인
- `outputs/acut_product_nvila8b_probe_v4/top_k_summary.json` 작업 확인
- `outputs/acut_product_nvila8b_probe_v5/top_k_summary.json` 작업 확인
- 외 66개 파일 변동

### Aesthetic Models
- `outputs/acut_product_nvila8b_probe_v3/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v10/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v5/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v6/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v7/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v8/top_k_summary.json` 작업 확인
- `outputs/acut_product_vila_local_score_reasoning_v9/top_k_summary.json` 작업 확인
- `outputs/acut_product_nvila8b_probe_v10_gpu/top_k_summary.json` 작업 확인
- `outputs/acut_product_nvila8b_probe_v4/top_k_summary.json` 작업 확인
- `outputs/acut_product_nvila8b_probe_v5/top_k_summary.json` 작업 확인
- 외 51개 파일 변동

### Technical Quality
- `configs/stage5_student_server.json` 작업 확인
- `outputs/test_vila_nvila_eval_20260415_214918/scores.csv` 작업 확인
- `outputs/test_vila_nvila_eval_20260415_214918/scores.jsonl` 작업 확인
- `outputs/test_vila_nvila_resume_20260415/ranked_results.csv` 작업 확인
- `outputs/test_vila_nvila_resume_20260415/ranked_results.jsonl` 작업 확인
- `outputs/acut_transparency_quick_export_check/top_k_summary.json` 작업 확인
- `outputs/acut_transparency_test_vila_20260416/app_results.csv` 작업 확인
- `outputs/acut_transparency_test_vila_20260416/app_results.json` 작업 확인
- `outputs/acut_transparency_test_vila_20260416/ranked_results.csv` 작업 확인
- `outputs/acut_transparency_test_vila_20260416/ranked_results.jsonl` 작업 확인
- 외 20개 파일 변동

### Firebase/Backend
- `outputs/firebase_worker_new_path_verify/top_k_summary.json` 작업 확인
- `outputs/firebase_worker_new_path_verify_gpu/top_k_summary.json` 작업 확인
- `outputs/firebase_worker_normal_path_verify/top_k_summary.json` 작업 확인
- `outputs/test_vila_worker_verify_full_20260416/top_k_summary.json` 작업 확인
- `outputs/test_vila_worker_verify_top10_20260416/top_k_summary.json` 작업 확인

### Mobile/Deployment
- `outputs/test_vila_nvila_resume_20260415/explanation_records.jsonl` 작업 확인
- `outputs/test_vila_nvila_resume_20260415/ranked_results.csv` 작업 확인
- `outputs/test_vila_nvila_resume_20260415/ranked_results.jsonl` 작업 확인
- `outputs/acut_transparency_test_vila_20260416/explanation_records.jsonl` 작업 확인
- `outputs/acut_transparency_test_vila_20260416/ranked_results.csv` 작업 확인
- `outputs/acut_transparency_test_vila_20260416/ranked_results.jsonl` 작업 확인
- `outputs/acut_transparency_test_vila_20260416_gpu/explanation_records.jsonl` 작업 확인
- `outputs/acut_transparency_test_vila_20260416_gpu/ranked_results.csv` 작업 확인
- `outputs/acut_transparency_test_vila_20260416_gpu/ranked_results.jsonl` 작업 확인
- `outputs/test_vila_worker_verify_full_20260416/explanation_records.jsonl` 작업 확인
- 외 5개 파일 변동

## 3. 실험 및 구현 내용

NVILA-8B 모델을 활용한 시각적 설명 생성 및 VILA-lite 핸드오프 준비.

## 4. 결과

- NVILA-8B 로컬 워커 기반 설명 생성 성공 (EVE_1437)
- VILA-lite 핸드오프 문서 작성 (EVE_0073)

## 5. 실패 / 문제 / 보류 사항

- NVILA 실행 시 GPU 메모리 제약 발생 가능성 (추정)

## 6. 의사결정

- 데이터 기반 성능 고도화 지속.

## 7. 산출물

- `configs/stage5_student_server.json`
- `outputs/acut_product_nvila8b_probe_v10_gpu/top_k_summary.json`
- `outputs/acut_product_nvila8b_probe_v3/top_k_summary.json`
- `outputs/acut_product_nvila8b_probe_v4/top_k_summary.json`
- `outputs/acut_product_nvila8b_probe_v5/top_k_summary.json`
- `outputs/acut_product_nvila8b_probe_v6/top_k_summary.json`
- `outputs/acut_product_nvila8b_probe_v7/top_k_summary.json`
- `outputs/acut_product_nvila8b_probe_v7_gpu/top_k_summary.json`
- `outputs/acut_product_nvila8b_probe_v8_gpu/top_k_summary.json`
- `outputs/acut_product_nvila8b_probe_v9_gpu/top_k_summary.json`
- `outputs/acut_product_vila_local_score_reasoning_v10/top_k_summary.json`
- `outputs/acut_product_vila_local_score_reasoning_v5/top_k_summary.json`
- `outputs/acut_product_vila_local_score_reasoning_v6/top_k_summary.json`
- `outputs/acut_product_vila_local_score_reasoning_v7/top_k_summary.json`
- `outputs/acut_product_vila_local_score_reasoning_v8/top_k_summary.json`

## 8. 다음 주 계획

경량 미적 품질 모델 확정 및 검증

## 9. 근거 목록

- [EVE_0044] outputs/acut_product_nvila8b_probe_v3/top_k_summary.json (날짜: 2026-04-13, 소스: content)
- [EVE_0045] outputs/acut_product_vila_local_score_reasoning_v10/top_k_summary.json (날짜: 2026-04-13, 소스: content)
- [EVE_0046] outputs/acut_product_vila_local_score_reasoning_v5/top_k_summary.json (날짜: 2026-04-13, 소스: content)
- [EVE_0047] outputs/acut_product_vila_local_score_reasoning_v6/top_k_summary.json (날짜: 2026-04-13, 소스: content)
- [EVE_0048] outputs/acut_product_vila_local_score_reasoning_v7/top_k_summary.json (날짜: 2026-04-13, 소스: content)
- [EVE_0049] outputs/acut_product_vila_local_score_reasoning_v8/top_k_summary.json (날짜: 2026-04-13, 소스: content)
- [EVE_0050] outputs/acut_product_vila_local_score_reasoning_v9/top_k_summary.json (날짜: 2026-04-13, 소스: content)
- [EVE_0051] outputs/acut_product_nvila8b_probe_v10_gpu/top_k_summary.json (날짜: 2026-04-14, 소스: content)
- [EVE_0052] outputs/acut_product_nvila8b_probe_v4/top_k_summary.json (날짜: 2026-04-14, 소스: content)
- [EVE_0053] outputs/acut_product_nvila8b_probe_v5/top_k_summary.json (날짜: 2026-04-14, 소스: content)
- [EVE_0054] outputs/acut_product_nvila8b_probe_v6/top_k_summary.json (날짜: 2026-04-14, 소스: content)
- [EVE_0055] outputs/acut_product_nvila8b_probe_v7/top_k_summary.json (날짜: 2026-04-14, 소스: content)
- [EVE_0056] outputs/acut_product_nvila8b_probe_v7_gpu/top_k_summary.json (날짜: 2026-04-14, 소스: content)
- [EVE_0057] outputs/acut_product_nvila8b_probe_v8_gpu/top_k_summary.json (날짜: 2026-04-14, 소스: content)
- [EVE_0058] outputs/acut_product_nvila8b_probe_v9_gpu/top_k_summary.json (날짜: 2026-04-14, 소스: content)
- [EVE_0059] configs/stage5_student_server.json (날짜: 2026-04-15, 소스: content)
- [EVE_0060] outputs/firebase_worker_new_path_verify/top_k_summary.json (날짜: 2026-04-15, 소스: content)
- [EVE_0061] outputs/firebase_worker_new_path_verify_gpu/top_k_summary.json (날짜: 2026-04-15, 소스: content)
- [EVE_0062] outputs/firebase_worker_normal_path_verify/top_k_summary.json (날짜: 2026-04-15, 소스: content)
- [EVE_0063] outputs/test_vila_nvila_eval_20260415_214918/scores.csv (날짜: 2026-04-15, 소스: filename)
- 외 57개 근거 파일 존재
# 4주차 연구일지: 2026-04-20 ~ 2026-04-26

## 1. 주차 목표

경량 미적 품질 모델 확정 및 검증

## 2. 수행 작업

### Aesthetic Models
- `configs/student_distill_conservative_full_aadb_20260423.json` 작업 확인
- `outputs/distill_conservative_aesthetic_full_aadb_20260423/resolved_teacher_config.json` 작업 확인
- `outputs/distill_conservative_aesthetic_full_aadb_20260423/summary.json` 작업 확인
- `outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_train.csv` 작업 확인
- `outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_trainval.csv` 작업 확인
- `outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_val.csv` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/manifest_val_baseline_disagreements.csv` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/manifest_val_scores.csv` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/manifest_val_student_disagreements.csv` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/summary.json` 작업 확인
- 외 75개 파일 변동

### VLM/Reasoning
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/summary.json` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/test_vila_baseline_disagreements.csv` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/test_vila_scores.csv` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/test_vila_student_disagreements.csv` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.verify_test_vila.json` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/handoff.md` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/disagreement_cases.csv` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/disagreement_cases.json` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/experiment_config.json` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/ranked_ensemble_experiment.csv` 작업 확인
- 외 62개 파일 변동

### Firebase/Backend
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/summary.json` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.verify_test_vila.json` 작업 확인

### Mobile/Deployment
- `outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.metadata.json` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.verify_test_vila.json` 작업 확인
- `outputs/student_conservative_aesthetic_full_aadb_20260423/handoff.md` 작업 확인

### Technical Quality
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/disagreement_cases.csv` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/disagreement_cases.json` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/experiment_config.json` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/ranked_ensemble_experiment.csv` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/ranked_ensemble_experiment.json` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/ranked_stage5_current.csv` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/ranked_stage5_current.json` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/raw_bundle_scores.csv` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/raw_bundle_scores.jsonl` 작업 확인
- `outputs/test_vila_aesthetic_ensemble_experiment_20260423/scores_with_ensemble.csv` 작업 확인
- 외 7개 파일 변동

## 3. 실험 및 구현 내용

Stage 5 학생 모델(MobileNetV2)의 최종 검증 및 TFLite 변환 준비.

## 4. 결과

- AADB val 기준 Pairwise Accuracy 90.56% 확보 (EVE_0041)
- MobileNetV2 경량화 모델 크기 1.7MB 달성

## 5. 실패 / 문제 / 보류 사항

- test_vila 기준 순위 보존 성능이 구형 모델 대비 압도적이지 않음 (EVE_0041)

## 6. 의사결정

- Stage 5 학생 모델을 실용적 캡스톤의 주력 미적 스코어러로 채택.

## 7. 산출물

- `configs/student_distill_conservative_full_aadb_20260423.json`
- `outputs/distill_conservative_aesthetic_full_aadb_20260423/resolved_teacher_config.json`
- `outputs/distill_conservative_aesthetic_full_aadb_20260423/summary.json`
- `outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_train.csv`
- `outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_trainval.csv`
- `outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_val.csv`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/manifest_val_baseline_disagreements.csv`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/manifest_val_scores.csv`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/manifest_val_student_disagreements.csv`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/summary.json`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/test_vila_baseline_disagreements.csv`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/test_vila_scores.csv`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/test_vila_student_disagreements.csv`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.metadata.json`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.verify_test_vila.json`

## 8. 다음 주 계획

Gemma 어댑터 학습 및 모바일 수출 조사

## 9. 근거 목록

- [EVE_0121] configs/student_distill_conservative_full_aadb_20260423.json (날짜: 2026-04-23, 소스: filename)
- [EVE_0122] outputs/distill_conservative_aesthetic_full_aadb_20260423/resolved_teacher_config.json (날짜: 2026-04-23, 소스: filename)
- [EVE_0123] outputs/distill_conservative_aesthetic_full_aadb_20260423/summary.json (날짜: 2026-04-23, 소스: filename)
- [EVE_0124] outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_train.csv (날짜: 2026-04-23, 소스: filename)
- [EVE_0125] outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_trainval.csv (날짜: 2026-04-23, 소스: filename)
- [EVE_0126] outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_val.csv (날짜: 2026-04-23, 소스: filename)
- [EVE_0127] outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/manifest_val_baseline_disagreements.csv (날짜: 2026-04-23, 소스: filename)
- [EVE_0128] outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/manifest_val_scores.csv (날짜: 2026-04-23, 소스: filename)
- [EVE_0129] outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/manifest_val_student_disagreements.csv (날짜: 2026-04-23, 소스: filename)
- [EVE_0130] outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/summary.json (날짜: 2026-04-23, 소스: filename)
- [EVE_0131] outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/test_vila_baseline_disagreements.csv (날짜: 2026-04-23, 소스: filename)
- [EVE_0132] outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/test_vila_scores.csv (날짜: 2026-04-23, 소스: filename)
- [EVE_0133] outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/test_vila_student_disagreements.csv (날짜: 2026-04-23, 소스: filename)
- [EVE_0134] outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.metadata.json (날짜: 2026-04-23, 소스: filename)
- [EVE_0135] outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.verify_test_vila.json (날짜: 2026-04-23, 소스: filename)
- [EVE_0136] outputs/student_conservative_aesthetic_full_aadb_20260423/handoff.md (날짜: 2026-04-23, 소스: filename)
- [EVE_0137] outputs/test_vila_aesthetic_ensemble_experiment_20260423/disagreement_cases.csv (날짜: 2026-04-23, 소스: filename)
- [EVE_0138] outputs/test_vila_aesthetic_ensemble_experiment_20260423/disagreement_cases.json (날짜: 2026-04-23, 소스: filename)
- [EVE_0139] outputs/test_vila_aesthetic_ensemble_experiment_20260423/experiment_config.json (날짜: 2026-04-23, 소스: filename)
- [EVE_0140] outputs/test_vila_aesthetic_ensemble_experiment_20260423/ranked_ensemble_experiment.csv (날짜: 2026-04-23, 소스: filename)
- 외 67개 근거 파일 존재
# 5주차 연구일지: 2026-04-27 ~ 2026-05-03

## 1. 주차 목표

Gemma 어댑터 학습 및 모바일 수출 조사

## 2. 수행 작업

### Gemma/LiteRT
- `outputs/gemini_api_teacher_aesthetic3_20260427/candidate_manifests_summary.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/scored_pool_inspection.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/test_vila.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/GEMMA_282_HANDOFF.md` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/dataset_all.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/dataset_all.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/export_gemma4_282/export_summary.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/README.md` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/checkpoint-80/README.md` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/checkpoint-80/adapter_config.json` 작업 확인
- 외 86개 파일 변동

### VLM/Reasoning
- `outputs/gemini_api_teacher_aesthetic3_20260427/candidate_manifests_summary.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/clean_teacher_dataset.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/clean_teacher_dataset.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/gemini_teacher_generation.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/gemini_teacher_generation_summary.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/gemma_manifest_summary.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/rejected_teacher_rows.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/review_summary.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/stage_a_300_candidate_manifest.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/stage_a_300_candidate_manifest.jsonl` 작업 확인
- 외 193개 파일 변동

### Aesthetic Models
- `outputs/gemini_api_teacher_aesthetic3_20260427/candidate_manifests_summary.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/clean_teacher_dataset.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/clean_teacher_dataset.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/gemini_teacher_generation.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/rejected_teacher_rows.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/scored_pool_inspection.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/stage_a_300_candidate_manifest.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/stage_a_300_candidate_manifest.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/stage_b_1000_candidate_manifest.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/stage_b_1000_candidate_manifest.jsonl` 작업 확인
- 외 68개 파일 변동

### Mobile/Deployment
- `outputs/gemini_api_teacher_aesthetic3_20260427/gemini_teacher_generation.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427/rejected_teacher_rows.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemini_teacher_generation_stage_a_300_20260427.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/checkpoint-80/processor_config.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/checkpoint-80/tokenizer.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/checkpoint-84/processor_config.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/checkpoint-84/tokenizer.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/final_adapter/processor_config.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/final_adapter/tokenizer.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/hf_cache/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c/processor_config.json` 작업 확인
- 외 56개 파일 변동

### Firebase/Backend
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/clean_teacher_dataset.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/clean_teacher_dataset.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/dataset_all.csv` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/dataset_all.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemini_teacher_generation_stage_a_300_20260427.jsonl` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/checkpoint-80/tokenizer.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/checkpoint-84/tokenizer.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/final_adapter/tokenizer.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/hf_cache/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c/tokenizer.json` 작업 확인
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/rejected_teacher_rows.csv` 작업 확인
- 외 42개 파일 변동

### Technical Quality
- `outputs/gemini_api_teacher_aesthetic3_highspec_100_batch_20260427/gemini_batch_requests_highspec_100_20260427.jsonl` 작업 확인

## 3. 실험 및 구현 내용

Gemma-4-E4B-it 대상 QLoRA 학습 및 JSON 출력 규격 학습.

## 4. 결과

- Gemma 282개 샘플 기반 JSON 스키마 학습 성공 (EVE_0047)
- 모바일 수출용 양자화 프로브 수행 (EVE_0344)

## 5. 실패 / 문제 / 보류 사항

- 시각적 근거 부족(Visual Hallucination)으로 인한 모바일 배포 부적격 판정 (EVE_0047)

## 6. 의사결정

- Gemma 어댑터의 시각적 환각 문제로 인해 실전 배포 보류.

## 7. 산출물

- `outputs/batch_status.json`
- `outputs/gemini_api_teacher_aesthetic3_20260427/candidate_manifests_summary.json`
- `outputs/gemini_api_teacher_aesthetic3_20260427/clean_teacher_dataset.csv`
- `outputs/gemini_api_teacher_aesthetic3_20260427/gemini_teacher_generation_summary.json`
- `outputs/gemini_api_teacher_aesthetic3_20260427/gemma_manifest_summary.json`
- `outputs/gemini_api_teacher_aesthetic3_20260427/rejected_teacher_rows.csv`
- `outputs/gemini_api_teacher_aesthetic3_20260427/review_summary.json`
- `outputs/gemini_api_teacher_aesthetic3_20260427/scored_pool_inspection.json`
- `outputs/gemini_api_teacher_aesthetic3_20260427/stage_a_300_candidate_manifest.csv`
- `outputs/gemini_api_teacher_aesthetic3_20260427/stage_b_1000_candidate_manifest.csv`
- `outputs/gemini_api_teacher_aesthetic3_20260427/stage_c_3000_candidate_manifest.csv`
- `outputs/gemini_api_teacher_aesthetic3_20260427/teacher_audit_rows.csv`
- `outputs/gemini_api_teacher_aesthetic3_20260427/teacher_audit_summary.json`
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/GEMMA_282_HANDOFF.md`
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/clean_teacher_dataset.csv`

## 8. 다음 주 계획

RGNet 및 A-LAMP 베이스라인 수립

## 9. 근거 목록

- [EVE_0208] outputs/gemini_api_teacher_aesthetic3_20260427/candidate_manifests_summary.json (날짜: 2026-04-27, 소스: filename)
- [EVE_0209] outputs/gemini_api_teacher_aesthetic3_20260427/clean_teacher_dataset.csv (날짜: 2026-04-27, 소스: filename)
- [EVE_0210] outputs/gemini_api_teacher_aesthetic3_20260427/clean_teacher_dataset.jsonl (날짜: 2026-04-27, 소스: filename)
- [EVE_0211] outputs/gemini_api_teacher_aesthetic3_20260427/gemini_teacher_generation.jsonl (날짜: 2026-04-27, 소스: filename)
- [EVE_0212] outputs/gemini_api_teacher_aesthetic3_20260427/gemini_teacher_generation_summary.json (날짜: 2026-04-27, 소스: filename)
- [EVE_0213] outputs/gemini_api_teacher_aesthetic3_20260427/gemma_manifest_summary.json (날짜: 2026-04-27, 소스: filename)
- [EVE_0214] outputs/gemini_api_teacher_aesthetic3_20260427/rejected_teacher_rows.csv (날짜: 2026-04-27, 소스: filename)
- [EVE_0215] outputs/gemini_api_teacher_aesthetic3_20260427/review_summary.json (날짜: 2026-04-27, 소스: filename)
- [EVE_0216] outputs/gemini_api_teacher_aesthetic3_20260427/scored_pool_inspection.json (날짜: 2026-04-27, 소스: filename)
- [EVE_0217] outputs/gemini_api_teacher_aesthetic3_20260427/stage_a_300_candidate_manifest.csv (날짜: 2026-04-27, 소스: filename)
- [EVE_0218] outputs/gemini_api_teacher_aesthetic3_20260427/stage_a_300_candidate_manifest.jsonl (날짜: 2026-04-27, 소스: filename)
- [EVE_0219] outputs/gemini_api_teacher_aesthetic3_20260427/stage_b_1000_candidate_manifest.csv (날짜: 2026-04-27, 소스: filename)
- [EVE_0220] outputs/gemini_api_teacher_aesthetic3_20260427/stage_b_1000_candidate_manifest.jsonl (날짜: 2026-04-27, 소스: filename)
- [EVE_0221] outputs/gemini_api_teacher_aesthetic3_20260427/stage_c_3000_candidate_manifest.csv (날짜: 2026-04-27, 소스: filename)
- [EVE_0222] outputs/gemini_api_teacher_aesthetic3_20260427/stage_c_3000_candidate_manifest.jsonl (날짜: 2026-04-27, 소스: filename)
- [EVE_0223] outputs/gemini_api_teacher_aesthetic3_20260427/teacher_audit_rows.csv (날짜: 2026-04-27, 소스: filename)
- [EVE_0224] outputs/gemini_api_teacher_aesthetic3_20260427/teacher_audit_summary.json (날짜: 2026-04-27, 소스: filename)
- [EVE_0225] outputs/gemini_api_teacher_aesthetic3_20260427/test_vila.jsonl (날짜: 2026-04-27, 소스: filename)
- [EVE_0226] outputs/gemini_api_teacher_aesthetic3_20260427_revised/GEMMA_282_HANDOFF.md (날짜: 2026-04-27, 소스: filename)
- [EVE_0227] outputs/gemini_api_teacher_aesthetic3_20260427_revised/clean_teacher_dataset.csv (날짜: 2026-04-27, 소스: filename)
- 외 238개 근거 파일 존재
# 6주차 연구일지: 2026-05-04 ~ 2026-05-10

## 1. 주차 목표

RGNet 및 A-LAMP 베이스라인 수립

## 2. 수행 작업

### Firebase/Backend
- `outputs/audit_aesthetic_model_origin_20260504/audit_report.md` 작업 확인
- `outputs/audit_model_paper_basis_20260506/audit_report.md` 작업 확인
- `outputs/audit_model_paper_basis_20260506/command_log.txt` 작업 확인
- `outputs/audit_model_paper_basis_20260506/gemini_review.md` 작업 확인
- `outputs/ava_pretrain_aadb_finetune_20260507/command_log.txt` 작업 확인
- `outputs/rgnet_float32_retry_20260508/command_log.txt` 작업 확인
- `outputs/color_aesthetics_audit_20260509/color_aesthetics_audit_report.md` 작업 확인
- `outputs/rgnet_paper_ava_classification_20260510/environment_check.json` 작업 확인

### Aesthetic Models
- `outputs/audit_aesthetic_model_origin_20260504/audit_report.md` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/old_three_model_scores.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/old_three_model_scores.jsonl` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/app_results.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/app_results.json` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/product_ranking.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/product_ranking.jsonl` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/ranked_results.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/ranked_results.jsonl` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/ranking_summary.json` 작업 확인
- 외 195개 파일 변동

### Mobile/Deployment
- `outputs/audit_aesthetic_model_origin_20260504/audit_report.md` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/summary.json` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/test_vila_aesthetic_comparison.csv` 작업 확인
- `outputs/audit_model_paper_basis_20260506/audit_report.md` 작업 확인
- `outputs/audit_model_paper_basis_20260506/command_log.txt` 작업 확인
- `outputs/audit_model_paper_basis_20260506/gemini_review.md` 작업 확인
- `outputs/audit_model_paper_basis_20260506/model_inventory.json` 작업 확인
- `outputs/ava_retrain_rgnet_alamp_20260506/ava_retrain_report.md` 작업 확인
- `outputs/ava_retrain_rgnet_alamp_20260506/command_log.txt` 작업 확인
- `outputs/ava_retrain_rgnet_alamp_20260506/metrics_summary.json` 작업 확인
- 외 67개 파일 변동

### VLM/Reasoning
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/old_three_model_scores.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/old_three_model_scores.jsonl` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/app_results.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/app_results.json` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/duplicate_groups.json` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/explanation_comparison.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/explanation_records.jsonl` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/product_ranking.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/product_ranking.jsonl` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/ranked_results.csv` 작업 확인
- 외 27개 파일 변동

### Technical Quality
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/old_three_model_scores.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/app_results.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/app_results.json` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/product_ranking.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/product_ranking.jsonl` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/ranked_results.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/ranked_results.jsonl` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/review_sheet.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/scores.csv` 작업 확인
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/scores.jsonl` 작업 확인
- 외 9개 파일 변동

### Gemma/LiteRT
- `outputs/test_vila_aesthetic_direct_compare_20260504_135316/summary.json` 작업 확인
- `outputs/audit_model_paper_basis_20260506/audit_report.md` 작업 확인
- `outputs/audit_model_paper_basis_20260506/command_log.txt` 작업 확인
- `outputs/ava_pretrain_aadb_finetune_20260507/command_log.txt` 작업 확인
- `outputs/rgnet_float32_retry_20260508/command_log.txt` 작업 확인
- `outputs/rgnet_paper_ava_classification_20260510/environment_check.json` 작업 확인

## 3. 실험 및 구현 내용

논문 구조를 모방한 미적 품질 평가 베이스라인 수립 및 전략 검토.

## 4. 결과

- RGNet AADB SRCC 0.6819 베이스라인 확보 (EVE_0042)
- A-LAMP Multi-Patch 구조 검토 및 학습 파이프라인 구축

## 5. 실패 / 문제 / 보류 사항

- 근거 파일 미확인

## 6. 의사결정

- 데이터 기반 성능 고도화 지속.

## 7. 산출물

- `outputs/aesthetic_weight_lab_structure_audit_20260508/file_inventory.json`
- `outputs/aesthetic_weight_lab_structure_audit_20260508/structure_audit_report.md`
- `outputs/aesthetic_weight_lab_tflite_tool_20260508/alamp_finetuned_smoke/raw_scores.csv`
- `outputs/aesthetic_weight_lab_tflite_tool_20260508/alamp_finetuned_smoke/raw_scores.json`
- `outputs/aesthetic_weight_lab_tflite_tool_20260508/alamp_finetuned_smoke/weight_presets.json`
- `outputs/aesthetic_weight_lab_tflite_tool_20260508/baseline_smoke/raw_scores.csv`
- `outputs/aesthetic_weight_lab_tflite_tool_20260508/baseline_smoke/raw_scores.json`
- `outputs/aesthetic_weight_lab_tflite_tool_20260508/baseline_smoke/weight_presets.json`
- `outputs/aesthetic_weight_lab_tflite_tool_20260508/handoff_report.md`
- `outputs/audit_aesthetic_model_origin_20260504/audit_report.md`
- `outputs/audit_model_paper_basis_20260506/audit_report.md`
- `outputs/audit_model_paper_basis_20260506/gemini_review.md`
- `outputs/audit_model_paper_basis_20260506/model_inventory.json`
- `outputs/ava_pretrain_aadb_finetune_20260507/alamp_ava_pretrain_aadb_finetune/train_summary.json`
- `outputs/ava_pretrain_aadb_finetune_20260507/checklist.md`

## 8. 다음 주 계획

그래프 기반 레이아웃 퓨전 및 ICAA TF 구현

## 9. 근거 목록

- [EVE_0466] outputs/audit_aesthetic_model_origin_20260504/audit_report.md (날짜: 2026-05-04, 소스: filename)
- [EVE_0467] outputs/test_vila_aesthetic_direct_compare_20260504_135316/old_three_model_scores.csv (날짜: 2026-05-04, 소스: filename)
- [EVE_0468] outputs/test_vila_aesthetic_direct_compare_20260504_135316/old_three_model_scores.jsonl (날짜: 2026-05-04, 소스: filename)
- [EVE_0469] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/app_results.csv (날짜: 2026-05-04, 소스: filename)
- [EVE_0470] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/app_results.json (날짜: 2026-05-04, 소스: filename)
- [EVE_0471] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/duplicate_groups.json (날짜: 2026-05-04, 소스: filename)
- [EVE_0472] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/explanation_comparison.csv (날짜: 2026-05-04, 소스: filename)
- [EVE_0473] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/explanation_records.jsonl (날짜: 2026-05-04, 소스: filename)
- [EVE_0474] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/product_ranking.csv (날짜: 2026-05-04, 소스: filename)
- [EVE_0475] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/product_ranking.jsonl (날짜: 2026-05-04, 소스: filename)
- [EVE_0476] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/ranked_results.csv (날짜: 2026-05-04, 소스: filename)
- [EVE_0477] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/ranked_results.jsonl (날짜: 2026-05-04, 소스: filename)
- [EVE_0478] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/ranking_summary.json (날짜: 2026-05-04, 소스: filename)
- [EVE_0479] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/review_sheet.csv (날짜: 2026-05-04, 소스: filename)
- [EVE_0480] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/review_topk.csv (날짜: 2026-05-04, 소스: filename)
- [EVE_0481] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/review_topk.md (날짜: 2026-05-04, 소스: filename)
- [EVE_0482] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/scores.csv (날짜: 2026-05-04, 소스: filename)
- [EVE_0483] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/scores.jsonl (날짜: 2026-05-04, 소스: filename)
- [EVE_0484] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/top_k.txt (날짜: 2026-05-04, 소스: filename)
- [EVE_0485] outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run/top_k_summary.json (날짜: 2026-05-04, 소스: filename)
- 외 191개 근거 파일 존재
# 7주차 연구일지: 2026-05-11 ~ 2026-05-17

## 1. 주차 목표

그래프 기반 레이아웃 퓨전 및 ICAA TF 구현

## 2. 수행 작업

### Aesthetic Models
- `outputs/alamp_detector_selection_20260511/command_log.txt` 작업 확인
- `outputs/alamp_detector_selection_20260511/detector_selection_report.md` 작업 확인
- `outputs/alamp_external_repo_audit_20260511/alamp_external_repo_audit_report.md` 작업 확인
- `outputs/alamp_external_repo_audit_20260511/command_log.txt` 작업 확인
- `outputs/alamp_object_graph_subset_20260511/context-notes.md` 작업 확인
- `outputs/alamp_object_graph_subset_20260511/detections/detection_summary.json` 작업 확인
- `outputs/alamp_object_graph_subset_20260511/detections_conf010/detection_summary.json` 작업 확인
- `outputs/alamp_object_graph_subset_20260511/detections_conf015/detection_summary.json` 작업 확인
- `outputs/alamp_object_graph_subset_20260511/graphs/graph_summary.json` 작업 확인
- `outputs/alamp_object_graph_subset_20260511/graphs_conf010/graph_summary.json` 작업 확인
- 외 527개 파일 변동

### Mobile/Deployment
- `outputs/alamp_paper_ava_classification_20260511/alamp_paper_ava_report.md` 작업 확인
- `outputs/alamp_paper_ava_classification_20260511/alamp_paper_ava_summary.json` 작업 확인
- `outputs/alamp_paper_ava_classification_20260511/command_log.txt` 작업 확인
- `outputs/alamp_paper_ava_classification_20260511/context-notes.md` 작업 확인
- `outputs/alamp_paper_ava_v1_design_audit_20260511/alamp_v1_design_audit_report.md` 작업 확인
- `outputs/alamp_paper_faithfulness_audit_20260511/alamp_gap_matrix.json` 작업 확인
- `outputs/alamp_paper_faithfulness_audit_20260511/alamp_paper_faithfulness_audit_report.md` 작업 확인
- `outputs/alamp_paper_mpnet_design_audit_20260511/alamp_paper_mpnet_design_audit_report.md` 작업 확인
- `outputs/alamp_paper_mpnet_design_audit_20260511/next_implementation_prompt.md` 작업 확인
- `outputs/alamp_paper_mpnet_design_audit_20260511/patch_selection_report.md` 작업 확인
- 외 329개 파일 변동

### Firebase/Backend
- `outputs/alamp_paper_ava_classification_20260511/context-notes.md` 작업 확인
- `outputs/icaa_export_safe_full_eval_20260514_190201/eval_log.txt` 작업 확인
- `outputs/icaa_export_safe_full_eval_20260514_190201/metrics.json` 작업 확인
- `outputs/icaa_export_safe_full_eval_20260514_193638/eval_log.txt` 작업 확인
- `outputs/icaa_export_safe_full_eval_20260514_193638/metrics.json` 작업 확인
- `outputs/icaa_export_safe_full_eval_20260514_193638/predictions.csv` 작업 확인
- `outputs/icaa_export_safe_onnx_20260514_194438/onnx_export_report.json` 작업 확인
- `outputs/icaa_export_safe_onnx_20260514_194704/onnx_export_report.json` 작업 확인
- `outputs/icaa_export_safe_onnx_20260514_194833/onnx_export_report.json` 작업 확인
- `outputs/icaa_export_safe_onnx_20260514_195608/onnx_export_report.json` 작업 확인
- 외 59개 파일 변동

### VLM/Reasoning
- `outputs/alamp_paper_ava_v1_design_audit_20260511/alamp_v1_design_decision.json` 작업 확인
- `outputs/alamp_paper_faithfulness_audit_20260511/alamp_paper_faithfulness_audit_report.md` 작업 확인
- `outputs/alamp_paper_faithfulness_audit_20260511/command_log.txt` 작업 확인
- `outputs/alamp_paper_mpnet_design_audit_20260511/alamp_paper_mpnet_decision.json` 작업 확인
- `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/manual_gemini_overlay_eval_report.md` 작업 확인
- `outputs/alamp_salient_detector_audit_20260512/salient_detector_audit_report.md` 작업 확인
- `outputs/alamp_salient_detector_audit_20260512/salient_detector_decision.json` 작업 확인

### Gemma/LiteRT
- `outputs/rgnet_old_vs_paper_fp16_compare_20260514/report.md` 작업 확인

### Technical Quality
- `outputs/eval_full_topiq_lite_spaq_replacement_20260517/artifacts.txt` 작업 확인
- `outputs/eval_full_topiq_lite_spaq_replacement_20260517/commands.txt` 작업 확인
- `outputs/eval_full_topiq_lite_spaq_replacement_20260517/report.md` 작업 확인
- `outputs/eval_full_topiq_lite_spaq_replacement_20260517/summary.csv` 작업 확인
- `outputs/eval_topiq_lite_flive_calibration_replacement_20260517/artifacts.txt` 작업 확인
- `outputs/eval_topiq_lite_flive_calibration_replacement_20260517/commands.txt` 작업 확인
- `outputs/eval_topiq_lite_flive_calibration_replacement_20260517/predictions_flive_test_mobile_flive.csv` 작업 확인
- `outputs/eval_topiq_lite_flive_calibration_replacement_20260517/predictions_flive_test_mobile_koniq.csv` 작업 확인
- `outputs/eval_topiq_lite_flive_calibration_replacement_20260517/predictions_flive_test_topiq_flive_calib.csv` 작업 확인
- `outputs/eval_topiq_lite_flive_calibration_replacement_20260517/predictions_flive_test_topiq_unified_orig.csv` 작업 확인
- 외 144개 파일 변동

## 3. 실험 및 구현 내용

A-LAMP 레이아웃 퓨전 실험 및 ICAA 모델의 TensorFlow 재구현.

## 4. 결과

- ICAA PyTorch 가중치와 TF Native 간 파리티(Parity) 확보 (EVE_1109)
- A-LAMP 4096 서브셋 대상 그래프 퓨전 시도

## 5. 실패 / 문제 / 보류 사항

- 그래프 퓨전 모델의 심각한 과적합 발생으로 인한 중단 결정 (EVE_0040)

## 6. 의사결정

- A-LAMP 레이아웃 퓨전 대신 Multi-Patch Teacher 개선에 집중하기로 함.

## 7. 산출물

- `outputs/alamp_detector_selection_20260511/detector_selection_report.md`
- `outputs/alamp_external_repo_audit_20260511/alamp_external_repo_audit_report.md`
- `outputs/alamp_mpnet_vs_aadb_alamp_review_20260513/aadb_alamp_metric_sources.json`
- `outputs/alamp_mpnet_vs_aadb_alamp_review_20260513/alamp_mpnet_vs_aadb_alamp_report.md`
- `outputs/alamp_mpnet_vs_aadb_alamp_review_20260513/alamp_mpnet_vs_aadb_alamp_summary.json`
- `outputs/alamp_object_graph_subset_20260511/context-notes.md`
- `outputs/alamp_object_graph_subset_20260511/detections/detection_summary.json`
- `outputs/alamp_object_graph_subset_20260511/detections_conf010/detection_summary.json`
- `outputs/alamp_object_graph_subset_20260511/detections_conf015/detection_summary.json`
- `outputs/alamp_object_graph_subset_20260511/graphs/graph_summary.json`
- `outputs/alamp_object_graph_subset_20260511/graphs_conf010/graph_summary.json`
- `outputs/alamp_object_graph_subset_20260511/object_graph_subset_report.md`
- `outputs/alamp_object_graph_subset_20260511/object_graph_subset_summary.json`
- `outputs/alamp_paper_ava_classification_20260511/alamp_paper_ava_report.md`
- `outputs/alamp_paper_ava_classification_20260511/alamp_paper_ava_summary.json`

## 8. 다음 주 계획

TOPIQ-lite 벤치마크 및 DistortionGuard 개발

## 9. 근거 목록

- [EVE_0677] outputs/alamp_detector_selection_20260511/command_log.txt (날짜: 2026-05-11, 소스: filename)
- [EVE_0678] outputs/alamp_detector_selection_20260511/detector_selection_report.md (날짜: 2026-05-11, 소스: filename)
- [EVE_0679] outputs/alamp_external_repo_audit_20260511/alamp_external_repo_audit_report.md (날짜: 2026-05-11, 소스: filename)
- [EVE_0680] outputs/alamp_external_repo_audit_20260511/command_log.txt (날짜: 2026-05-11, 소스: filename)
- [EVE_0681] outputs/alamp_object_graph_subset_20260511/context-notes.md (날짜: 2026-05-11, 소스: filename)
- [EVE_0682] outputs/alamp_object_graph_subset_20260511/detections/detection_summary.json (날짜: 2026-05-11, 소스: filename)
- [EVE_0683] outputs/alamp_object_graph_subset_20260511/detections_conf010/detection_summary.json (날짜: 2026-05-11, 소스: filename)
- [EVE_0684] outputs/alamp_object_graph_subset_20260511/detections_conf015/detection_summary.json (날짜: 2026-05-11, 소스: filename)
- [EVE_0685] outputs/alamp_object_graph_subset_20260511/graphs/graph_summary.json (날짜: 2026-05-11, 소스: filename)
- [EVE_0686] outputs/alamp_object_graph_subset_20260511/graphs_conf010/graph_summary.json (날짜: 2026-05-11, 소스: filename)
- [EVE_0687] outputs/alamp_object_graph_subset_20260511/object_graph_subset_report.md (날짜: 2026-05-11, 소스: filename)
- [EVE_0688] outputs/alamp_object_graph_subset_20260511/object_graph_subset_summary.json (날짜: 2026-05-11, 소스: filename)
- [EVE_0689] outputs/alamp_paper_ava_classification_20260511/alamp_paper_ava_report.md (날짜: 2026-05-11, 소스: filename)
- [EVE_0690] outputs/alamp_paper_ava_classification_20260511/alamp_paper_ava_summary.json (날짜: 2026-05-11, 소스: filename)
- [EVE_0691] outputs/alamp_paper_ava_classification_20260511/checklist.md (날짜: 2026-05-11, 소스: filename)
- [EVE_0692] outputs/alamp_paper_ava_classification_20260511/command_log.txt (날짜: 2026-05-11, 소스: filename)
- [EVE_0693] outputs/alamp_paper_ava_classification_20260511/context-notes.md (날짜: 2026-05-11, 소스: filename)
- [EVE_0694] outputs/alamp_paper_ava_classification_20260511/forward_smoke.json (날짜: 2026-05-11, 소스: filename)
- [EVE_0695] outputs/alamp_paper_ava_classification_20260511/mid_eval/v0_a/evaluation_summary.json (날짜: 2026-05-11, 소스: filename)
- [EVE_0696] outputs/alamp_paper_ava_classification_20260511/mid_eval/v0_b_fixed/evaluation_summary.json (날짜: 2026-05-11, 소스: filename)
- 외 678개 근거 파일 존재
# 8주차 연구일지: 2026-05-18 ~ 2026-05-24

## 1. 주차 목표

TOPIQ-lite 벤치마크 및 DistortionGuard 개발

## 2. 수행 작업

### Aesthetic Models
- `outputs/mobile_alamp_v2_full_ava_20260519/best_model_eval_report.md` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519/prediction_summary.json` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519/report.md` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519/summary.json` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/export_metadata.json` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/export_metadata.json` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/parity_report.json` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/parity_report.md` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_report.json` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_report.md` 작업 확인
- 외 98개 파일 변동

### Mobile/Deployment
- `outputs/mobile_alamp_v2_full_ava_20260519/best_model_eval_report.md` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519/report.md` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/export_metadata.json` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/export_metadata.json` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/parity_report.json` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/parity_report.md` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_report.json` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_report.md` 작업 확인
- `outputs/mobile_alamp_v2_full_ava_smoke_20260519/report.md` 작업 확인
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/report.md` 작업 확인
- 외 26개 파일 변동

### Technical Quality
- `outputs/dryrun_techiqa_guard_v1_stage2_partial_unfreeze_20260520/training_summary.json` 작업 확인
- `outputs/dryrun_techiqa_guard_v1_stage4_fpdata_check_20260520/training_summary.json` 작업 확인
- `outputs/dryrun_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520/model_summary.txt` 작업 확인
- `outputs/dryrun_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520/training_log.csv` 작업 확인
- `outputs/dryrun_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520/training_summary.json` 작업 확인
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_existing_avg_technical_flive.csv` 작업 확인
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_existing_avg_technical_koniq.csv` 작업 확인
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_flive_image_mobile_flive.csv` 작업 확인
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_flive_image_mobile_koniq.csv` 작업 확인
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_koniq_mobile_flive.csv` 작업 확인
- 외 258개 파일 변동

### Firebase/Backend
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_existing_avg_technical_flive.csv` 작업 확인
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_flive_image_mobile_flive.csv` 작업 확인
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_koniq_mobile_flive.csv` 작업 확인
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_topiq_lite_mixed112_frozen_fp16_flive.csv` 작업 확인
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_topiq_lite_ranking_lam01_gap05_fp16_flive.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage2_partial_unfreeze_b3_flive15_20260520/predictions_existing_avg_technical_flive.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage2_partial_unfreeze_b3_flive15_20260520/predictions_flive_image_mobile_flive.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage2_partial_unfreeze_b3_flive15_20260520/predictions_koniq_mobile_flive.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage2_partial_unfreeze_b3_flive15_20260520/predictions_techiqa_guard_v1_s1_frozen_flive.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage2_partial_unfreeze_b3_flive15_20260520/predictions_topiq_lite_mixed112_frozen_fp16_flive.csv` 작업 확인
- 외 38개 파일 변동

### VLM/Reasoning
- `outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/predictions_existing_avg_technical_hard_fp.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/predictions_existing_avg_technical_smoke.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/predictions_flive_image_mobile_hard_fp.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/predictions_flive_image_mobile_smoke.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/predictions_koniq_mobile_hard_fp.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/predictions_koniq_mobile_smoke.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/predictions_techiqa_guard_v1_s1_frozen_hard_fp.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/predictions_techiqa_guard_v1_s1_frozen_smoke.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/predictions_topiq_lite_mixed112_frozen_fp16_hard_fp.csv` 작업 확인
- `outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/predictions_topiq_lite_mixed112_frozen_fp16_smoke.csv` 작업 확인
- 외 108개 파일 변동

## 3. 실험 및 구현 내용

최신 기술적 품질 모델 도입 및 다단계 왜곡 감지 모델 개발.

## 4. 결과

- TOPIQ-lite(mixed112) SPAQ SRCC 0.899 달성 (EVE_0046)
- DistortionGuard Stage A/B 학습 완료

## 5. 실패 / 문제 / 보류 사항

- DistortionGuard B3 버전이 FLIVE 데이터 편향으로 인해 기술적 결함 오탐지 증가 (EVE_0045)

## 6. 의사결정

- 기술적 품질 모델로 TOPIQ-lite(mixed112) 우선 고려.

## 7. 산출물

- `outputs/aesthetic_paper_benchmark_old_vs_new_20260523/report.md`
- `outputs/aesthetic_paper_benchmark_old_vs_new_20260523/summary.csv`
- `outputs/aesthetic_paper_benchmark_old_vs_new_20260523/summary.json`
- `outputs/aesthetic_paper_benchmark_old_vs_new_20260523/verification_report.md`
- `outputs/alamp_dual_branch_4096_comparison_20260524/checklist.md`
- `outputs/alamp_dual_branch_4096_comparison_20260524/context-notes.md`
- `outputs/alamp_dual_branch_4096_comparison_20260524/eval_dual_branch_gcn_4096/evaluation_summary.json`
- `outputs/alamp_dual_branch_4096_comparison_20260524/eval_multipatch_only_4096/report.md`
- `outputs/alamp_dual_branch_4096_comparison_20260524/eval_multipatch_only_4096/summary.json`
- `outputs/alamp_dual_branch_4096_comparison_20260524/matched_subset_summary.json`
- `outputs/alamp_dual_branch_4096_comparison_20260524/plan.md`
- `outputs/alamp_dual_branch_4096_comparison_20260524/report.md`
- `outputs/alamp_dual_branch_4096_comparison_20260524/summary.json`
- `outputs/alamp_dual_branch_gcn_failure_analysis_20260524/report.md`
- `outputs/alamp_dual_branch_gcn_teacher_preflight_20260524/report.md`

## 8. 다음 주 계획

NIMA 버그 수정 및 최종 성능 재검증

## 9. 근거 목록

- [EVE_1375] outputs/mobile_alamp_v2_full_ava_20260519/best_model_eval_report.md (날짜: 2026-05-19, 소스: filename)
- [EVE_1376] outputs/mobile_alamp_v2_full_ava_20260519/prediction_summary.json (날짜: 2026-05-19, 소스: filename)
- [EVE_1377] outputs/mobile_alamp_v2_full_ava_20260519/report.md (날짜: 2026-05-19, 소스: filename)
- [EVE_1378] outputs/mobile_alamp_v2_full_ava_20260519/summary.json (날짜: 2026-05-19, 소스: filename)
- [EVE_1379] outputs/mobile_alamp_v2_full_ava_20260519_tflite/export_metadata.json (날짜: 2026-05-19, 소스: filename)
- [EVE_1380] outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/export_metadata.json (날짜: 2026-05-19, 소스: filename)
- [EVE_1381] outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/parity_report.json (날짜: 2026-05-19, 소스: filename)
- [EVE_1382] outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/parity_report.md (날짜: 2026-05-19, 소스: filename)
- [EVE_1383] outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_report.json (날짜: 2026-05-19, 소스: filename)
- [EVE_1384] outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_report.md (날짜: 2026-05-19, 소스: filename)
- [EVE_1385] outputs/mobile_alamp_v2_full_ava_smoke_20260519/prediction_summary.json (날짜: 2026-05-19, 소스: filename)
- [EVE_1386] outputs/mobile_alamp_v2_full_ava_smoke_20260519/report.md (날짜: 2026-05-19, 소스: filename)
- [EVE_1387] outputs/mobile_alamp_v2_full_ava_smoke_20260519/summary.json (날짜: 2026-05-19, 소스: filename)
- [EVE_1388] outputs/dryrun_techiqa_guard_v1_stage2_partial_unfreeze_20260520/training_summary.json (날짜: 2026-05-20, 소스: filename)
- [EVE_1389] outputs/dryrun_techiqa_guard_v1_stage4_fpdata_check_20260520/training_summary.json (날짜: 2026-05-20, 소스: filename)
- [EVE_1390] outputs/dryrun_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520/model_summary.txt (날짜: 2026-05-20, 소스: filename)
- [EVE_1391] outputs/dryrun_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520/training_log.csv (날짜: 2026-05-20, 소스: filename)
- [EVE_1392] outputs/dryrun_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520/training_summary.json (날짜: 2026-05-20, 소스: filename)
- [EVE_1393] outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_existing_avg_technical_flive.csv (날짜: 2026-05-20, 소스: filename)
- [EVE_1394] outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_existing_avg_technical_koniq.csv (날짜: 2026-05-20, 소스: filename)
- 외 378개 근거 파일 존재
# 9주차 연구일지: 2026-05-25 ~ 2026-05-31

## 1. 주차 목표

NIMA 버그 수정 및 최종 성능 재검증

## 2. 수행 작업

### Aesthetic Models
- `outputs/alamp_papercap_recipe_audit_20260525/report.md` 작업 확인
- `outputs/alamp_patch_fidelity_audit_20260525/report.md` 작업 확인
- `outputs/dryrun_alamp_multipatch_teacher_default_compat_20260525/train_summary.json` 작업 확인
- `outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525/context-notes.md` 작업 확인
- `outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525/plan.md` 작업 확인
- `outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525/train_summary.json` 작업 확인
- `outputs/dryrun_alamp_multipatch_teacher_papercap4096_recipe_20260525/context-notes.md` 작업 확인
- `outputs/dryrun_alamp_multipatch_teacher_papercap4096_recipe_20260525/plan.md` 작업 확인
- `outputs/dryrun_alamp_multipatch_teacher_papercap4096_recipe_20260525/train_summary.json` 작업 확인
- `outputs/dryrun_alamp_multipatch_teacher_papercap4096_sgd_20260525/context-notes.md` 작업 확인
- 외 84개 파일 변동

### Mobile/Deployment
- `outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525/plan.md` 작업 확인
- `outputs/icaa_performance_reverification_20260525/metrics_summary.json` 작업 확인
- `outputs/icaa_performance_reverification_20260525/model_inventory.json` 작업 확인
- `outputs/icaa_performance_reverification_20260525/parity_summary.json` 작업 확인
- `outputs/icaa_performance_reverification_20260525/report.md` 작업 확인
- `outputs/nima_fixed_preproc_eval_20260525/nima_mobile_fixed_preproc_fp32.metadata.json` 작업 확인
- `outputs/nima_fixed_preproc_eval_20260525/report.md` 작업 확인
- `outputs/nima_icaa_paper_level_eval_20260525/metrics_summary.json` 작업 확인
- `outputs/nima_icaa_paper_level_eval_20260525/nima_parity.json` 작업 확인
- `outputs/nima_icaa_paper_level_eval_20260525/report.md` 작업 확인
- 외 23개 파일 변동

### Firebase/Backend
- `outputs/icaa_performance_reverification_20260525/icaa_predictions.csv` 작업 확인
- `outputs/nima_preprocessing_fix_20260525/report.md` 작업 확인
- `outputs/acut_weekly_research_journal_v2_20260526/parsed/file_inventory.txt` 작업 확인
- `outputs/acut_weekly_research_journal_v2_20260526/parsed/potential_evidence_files.txt` 작업 확인
- `outputs/rgnet_paper_reproduction_preflight_20260526/report.md` 작업 확인

### Technical Quality
- `outputs/nima_preprocessing_fix_20260525/report.md` 작업 확인
- `outputs/acut_capstone_final_report_20260526/evidence_index.md` 작업 확인
- `outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv` 작업 확인
- `outputs/acut_capstone_final_report_20260526/report.md` 작업 확인
- `outputs/acut_capstone_final_report_20260526/report_generation_log.md` 작업 확인
- `outputs/acut_weekly_research_journal_20260526/experiment_timeline.csv` 작업 확인
- `outputs/acut_weekly_research_journal_20260526/weekly_evidence_index.md` 작업 확인
- `outputs/acut_weekly_research_journal_20260526/weekly_research_journal.md` 작업 확인
- `outputs/acut_weekly_research_journal_20260526/weekly_summary.csv` 작업 확인
- `outputs/acut_weekly_research_journal_v2_20260526/parsed/file_inventory.txt` 작업 확인
- 외 2개 파일 변동

### VLM/Reasoning
- `outputs/acut_capstone_final_report_20260526/report.md` 작업 확인
- `outputs/acut_weekly_research_journal_20260526/experiment_timeline.csv` 작업 확인
- `outputs/acut_weekly_research_journal_20260526/report_generation_log.md` 작업 확인
- `outputs/acut_weekly_research_journal_20260526/weekly_research_journal.md` 작업 확인
- `outputs/acut_weekly_research_journal_20260526/weekly_summary.csv` 작업 확인
- `outputs/acut_weekly_research_journal_v2_20260526/parsed/file_inventory.txt` 작업 확인
- `outputs/acut_weekly_research_journal_v2_20260526/parsed/files_gemma_litert.txt` 작업 확인
- `outputs/acut_weekly_research_journal_v2_20260526/parsed/potential_evidence_files.txt` 작업 확인
- `outputs/rgnet_paper_reproduction_preflight_20260526/report.md` 작업 확인
- `outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526/report.md` 작업 확인

### Gemma/LiteRT
- `outputs/acut_weekly_research_journal_20260526/weekly_research_journal.md` 작업 확인
- `outputs/acut_weekly_research_journal_v2_20260526/parsed/file_inventory.txt` 작업 확인
- `outputs/acut_weekly_research_journal_v2_20260526/parsed/files_gemma_litert.txt` 작업 확인
- `outputs/acut_weekly_research_journal_v2_20260526/parsed/potential_evidence_files.txt` 작업 확인
- `outputs/rgnet_paper_reproduction_preflight_20260526/report.md` 작업 확인

## 3. 실험 및 구현 내용

NIMA 전처리 버그 수정 및 주요 모델의 TFLite 최종 패키징.

## 4. 결과

- NIMA EfficientNetV2 중복 전처리 수정 후 SRCC 0.59 달성 (EVE_0044)
- ICAA Color SRCC 0.8791 공식 성능 재현 확인 (EVE_0043)

## 5. 실패 / 문제 / 보류 사항

- RGNet DenseASPP ablation 결과가 베이스라인(SRCC 0.68)에 미달 (EVE_0042)

## 6. 의사결정

- 데이터 기반 성능 고도화 지속.

## 7. 산출물

- `outputs/acut_capstone_final_report_20260526/evidence_index.md`
- `outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`
- `outputs/acut_capstone_final_report_20260526/report.md`
- `outputs/acut_capstone_final_report_20260526/report_generation_log.md`
- `outputs/acut_weekly_research_journal_20260526/experiment_timeline.csv`
- `outputs/acut_weekly_research_journal_20260526/report_generation_log.md`
- `outputs/acut_weekly_research_journal_20260526/weekly_evidence_index.md`
- `outputs/acut_weekly_research_journal_20260526/weekly_research_journal.md`
- `outputs/acut_weekly_research_journal_20260526/weekly_summary.csv`
- `outputs/aesthetic_teacher_reproduction_decision_20260526/report.md`
- `outputs/alamp_papercap_recipe_audit_20260525/report.md`
- `outputs/alamp_patch_fidelity_audit_20260525/report.md`
- `outputs/dryrun_alamp_multipatch_teacher_default_compat_20260525/train_summary.json`
- `outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525/context-notes.md`
- `outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525/plan.md`

## 8. 다음 주 계획

캡스톤 최종 발표 및 결과물 정리

## 9. 근거 목록

- [EVE_1773] outputs/alamp_papercap_recipe_audit_20260525/report.md (날짜: 2026-05-25, 소스: filename)
- [EVE_1774] outputs/alamp_patch_fidelity_audit_20260525/report.md (날짜: 2026-05-25, 소스: filename)
- [EVE_1775] outputs/dryrun_alamp_multipatch_teacher_default_compat_20260525/train_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1776] outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525/context-notes.md (날짜: 2026-05-25, 소스: filename)
- [EVE_1777] outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525/plan.md (날짜: 2026-05-25, 소스: filename)
- [EVE_1778] outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525/train_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1779] outputs/dryrun_alamp_multipatch_teacher_papercap4096_recipe_20260525/context-notes.md (날짜: 2026-05-25, 소스: filename)
- [EVE_1780] outputs/dryrun_alamp_multipatch_teacher_papercap4096_recipe_20260525/plan.md (날짜: 2026-05-25, 소스: filename)
- [EVE_1781] outputs/dryrun_alamp_multipatch_teacher_papercap4096_recipe_20260525/train_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1782] outputs/dryrun_alamp_multipatch_teacher_papercap4096_sgd_20260525/context-notes.md (날짜: 2026-05-25, 소스: filename)
- [EVE_1783] outputs/dryrun_alamp_multipatch_teacher_papercap4096_sgd_20260525/plan.md (날짜: 2026-05-25, 소스: filename)
- [EVE_1784] outputs/dryrun_alamp_multipatch_teacher_papercap4096_sgd_20260525/train_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1785] outputs/dryrun_alamp_multipatch_teacher_recipe_default_compat_20260525/train_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1786] outputs/dryrun_alamp_multipatch_teacher_sgd_default_compat_20260525/train_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1787] outputs/dryrun_rgnet_v1_cascaded_denseaspp_aadb_20260525/train_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1788] outputs/dryrun_rgnet_v1_default_context_compat_20260525/train_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1789] outputs/dryrun_rgnet_v1_hybrid_spatial_20260525/train_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1790] outputs/dryrun_rgnet_v1_hybrid_spatial_default_compat_20260525/train_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1791] outputs/eval_rgnet_v1_cascaded_denseaspp_aadb_full_20260525/evaluation_summary.json (날짜: 2026-05-25, 소스: filename)
- [EVE_1792] outputs/icaa_performance_reverification_20260525/icaa_predictions.csv (날짜: 2026-05-25, 소스: filename)
- 외 77개 근거 파일 존재
