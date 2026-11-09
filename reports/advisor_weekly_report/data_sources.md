# 시각자료 데이터 출처 (Data Sources)

| 생성물 | 원본 데이터 소스 | 파싱 및 가공 방법 |
| :--- | :--- | :--- |
| `tables/experiment_timeline.csv` | `outputs/acut_weekly_research_journal_v2_20260526/weekly_research_journal_combined.md` | 주차별 목표와 주요 수행 작업 텍스트를 분석하여 9개 주차의 핵심 마일스톤을 요약 추출. |
| `tables/model_inventory.csv` | `outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`, 각 모델별 폴더 내 `report.md` | 모델명, 데이터셋, 성능 지표 추출 후, 실패/보류/사용 여부를 텍스트 리포트 기반으로 판단하여 하드코딩 맵핑. (특히 Stage 5는 지시사항에 따라 '사용 안 함'으로 명시 처리) |
| `tables/success_failure_matrix.csv` | 프로젝트 폴더 전체의 `report.md`, `summary.json`, `command_log.txt` 등 | 개별 실험의 의도와 최종 판단 결과를 매트릭스 형태로 정리. |
| `figures/timeline.png` | `tables/experiment_timeline.csv` | Matplotlib을 사용해 주차별 수평 막대 그래프 생성 |
| `figures/model_status.png` | `tables/model_inventory.csv` | 상태(Adopted, Failed 등) 개수를 세어 Pie Chart로 시각화. (한글 폰트 경고를 피하기 위해 영문 라벨 사용) |
| `figures/performance_srcc.png` | `tables/model_inventory.csv` | 출력값(Output)이 SRCC로 기재된 모델 필터링 후 Bar Chart 생성. |
