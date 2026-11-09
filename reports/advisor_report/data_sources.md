# 시각자료 데이터 출처 (Data Sources)

이 문서는 `advisor_report.md`에 포함된 각 시각자료와 표가 어떤 원본 데이터 파일에서 생성되었는지 기록합니다. 모든 시각자료는 실제 프로젝트 평가 및 벤치마크 로그를 기반으로 생성되었습니다.

### 1. B_model_summary.csv (B. 모델 구성 요약표)
- **출처 파일**: `/home/omen_pc1/photo_score_project/outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`
- **사용 방식**: 전체 모델 메트릭 요약본에서 모델명, 역할(category), 학습 데이터셋, TFLite 변환 여부, 비고(notes) 컬럼을 추출하여 구성.

### 2. C_technical_quality_comparison.png (C. 기술 품질 모델 비교 그래프)
- **출처 파일**: `/home/omen_pc1/photo_score_project/outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`
- **사용 방식**: `category == 'Technical'`인 행을 필터링한 후, `srcc` (Spearman Rank Correlation Coefficient) 지표를 추출하여 막대 그래프로 시각화. FLIVE, KonIQ, TOPIQ-lite 모델 데이터 사용.

### 3. D_aesthetic_quality_comparison.png (D. 미적 품질 모델 비교 그래프)
- **출처 파일**: `/home/omen_pc1/photo_score_project/outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`
- **사용 방식**: `category == 'Aesthetic'`인 행을 필터링. 모델별 평가지표가 SRCC와 Accuracy로 나뉘어 있으므로, 각각을 분리하여 Grouped/Subplot 형태로 시각화. NIMA, RGNet, A-LAMP, ICAA, Stage5 Student 모델 데이터 사용.

### 4. E_score_distribution.png (E. score distribution histogram)
- **출처 파일**: `/home/omen_pc1/photo_score_project/outputs/rgnet_paper_v1_aadb_regression_20260509/eval/test_predictions.csv`
- **사용 방식**: AADB 데이터셋에 대한 RGNet 베이스라인 모델의 실제 Ground Truth Score와 Predicted Score 분포를 Histogram으로 중첩 시각화.

### 5. F_prediction_scatter.png (F. prediction scatter plot)
- **출처 파일**: `/home/omen_pc1/photo_score_project/outputs/rgnet_paper_v1_aadb_regression_20260509/eval/test_predictions.csv`
- **사용 방식**: x축에 Ground Truth, y축에 Prediction 값을 매핑. 피어슨(PLCC) 및 스피어만(SRCC) 상관계수 계산 포함.

### 6. G_hard_case_grid.png (G. hard case image grid)
- **출처 파일 1 (데이터)**: `/home/omen_pc1/photo_score_project/outputs/rgnet_paper_v1_aadb_regression_20260509/eval/test_predictions.csv`
- **출처 파일 2 (이미지)**: `/home/omen_pc1/photo_score_project/data/raw/aadb/images/` 내부 JPEG 파일
- **사용 방식**: 예측 에러(`Prediction - Score`)가 가장 큰 2개 이미지(Hard False Positive)와 가장 작은 2개 이미지(Hard False Negative)의 실제 로컬 이미지 파일을 읽어와 2x2 Grid 형식으로 렌더링.

### 7. I_decision_matrix.csv (I. 최종 의사결정 matrix)
- **출처 파일**: `/home/omen_pc1/photo_score_project/outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`
- **사용 방식**: 성능 지표, TFLite (Mobile Ready) 상태, 기존 보고서 내 텍스트 노트("Failed", "Best practical" 등)를 파싱하여 의사결정(Adopted, Discarded, Hold) 상태를 부여한 매트릭스 도출.
