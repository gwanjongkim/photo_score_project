# 기존 보고서 감사 요약 (Audit Summary)

### 1. 감사 개요
- **목적**: 기존 A-cut 캡스톤 최종 보고서(`report.md`)가 파일 위치만 나열하는 수준인지 점검하고, 교수 보고용으로 적합하게 시각적 근거가 바로 노출될 수 있도록 개선.
- **감사 대상 경로**: `/home/omen_pc1/photo_score_project/outputs/acut_capstone_final_report_20260526/`

### 2. 발견된 문제점
1. **시각적 근거 부재**: 기존 `report.md` 본문에는 데이터 기반의 그래프(Histogram, Scatter 등)가 포함되어 있지 않았으며, 맨 하단 부록 챕터에 `figures/...` 경로만 단순 링크되어 있었습니다. 
2. **원시 데이터와의 괴리**: 모델의 강건성을 입증할 수 있는 실제 점수 분포도(Distribution)나 예측 오차 산점도(Scatter Plot)가 리포트에서 누락되어, 모델 신뢰도에 대한 직관적 이해가 어렵습니다.
3. **Hard Case 분석 미비**: 모델이 실패하거나 성공한 구체적 이미지(예: False Positive)와 모델 점수가 함께 표시된 시각적 증거가 부족하여 왜 특정 모델을 배제하거나 채택했는지에 대한 설득력이 부족합니다.

### 3. 개선 조치 (Action Taken)
- **데이터 직접 추출**: `model_metrics_summary.csv`를 파싱하여 정확한 수치 기반의 비교 막대그래프(C, D)를 Python 스크립트로 신규 생성.
- **실제 추론 결과 시각화**: `rgnet_paper_v1_aadb_regression_20260509/eval/test_predictions.csv` 등의 실제 Prediction 로그를 읽어와 점수 분포 히스토그램(E)과 산점도(F)를 렌더링.
- **실제 이미지 그리드 생성**: 예측 에러가 가장 큰 이미지(Hard Case)를 로컬 이미지 경로에서 직접 로드하여 모델 점수와 에러를 포함한 시각적 그리드(G) 생성.
- **보고서 재구성**: 위에서 생성된 그래프 및 CSV 표를 마크다운에 직접 삽입하여, 교수님이 보고서만 읽어도 모든 근거를 눈으로 확인할 수 있는 `advisor_report.md` 작성.
