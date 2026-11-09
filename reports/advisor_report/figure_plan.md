# 시각화 계획 및 상태 (Figure Plan)

### 생성 완료된 시각자료 목록

| ID | 요구사항 | 생성 여부 | 파일명 / 방식 | 비고 |
|:---|:---|:---:|:---|:---|
| **A** | 전체 A-cut 평가 파이프라인 구조도 | ✅ | `advisor_report.md` 내 Mermaid | 구조의 명확성을 위해 텍스트 기반 Mermaid 다이어그램 채택 |
| **B** | 모델 구성 요약표 | ✅ | `tables/B_model_summary.csv` | 원본 모델 요약본에서 주요 열 추출 |
| **C** | 기술 품질 모델 비교 그래프 | ✅ | `figures/C_technical_quality_comparison.png` | TOPIQ, FLIVE, KonIQ 비교 (SRCC 기준) |
| **D** | 미적 품질 모델 비교 그래프 | ✅ | `figures/D_aesthetic_quality_comparison.png` | Accuracy 그룹과 SRCC 그룹 분리 시각화 |
| **E** | Score distribution histogram | ✅ | `figures/E_score_distribution.png` | RGNet AADB 추론 CSV 활용 |
| **F** | Prediction scatter plot | ✅ | `figures/F_prediction_scatter.png` | RGNet AADB 추론 CSV 내 GT vs Pred |
| **G** | Hard case image grid | ✅ | `figures/G_hard_case_grid.png` | 최대/최소 에러 이미지 파일 실제 로드하여 생성 |
| **I** | 최종 의사결정 matrix | ✅ | `tables/I_decision_matrix.csv` | 성능 및 TFLite Export 기준으로 상태 자동 부여 |

### 생성하지 못한 시각자료

| ID | 요구사항 | 생성 여부 | 이유 | 대안 |
|:---|:---|:---:|:---|:---|
| **H** | Mobile deployment trade-off chart | ❌ 생성 불가 | `model_metrics_summary.csv` 등에 모델별 Inference Time (추론 속도) 및 구체적인 모델 파라미터 크기(MB) 데이터가 명시적으로 존재하지 않음. 텍스트 보고서 내부에 Stage 5 모델 크기(1.7MB)만 부분적으로 언급되어 있어, 객관적인 수치 기반 그래프 생성이 불가능함. | 의사결정 매트릭스(I)의 `Mobile Ready` 항목 및 최종 보고서 본문의 텍스트 설명으로 대체함. |
