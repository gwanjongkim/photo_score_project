# 시각화 계획 및 상태 (Figure Plan)

### 생성된 시각자료

1. **`figures/timeline.png`**: 주차별 주요 마일스톤과 프로젝트 진행 과정을 보여주는 가로형 Bar Chart.
2. **`figures/model_status.png`**: 검토된 전체 모델 중 Adopted(사용), Discarded(사용 안 함/폐기), Hold(보류), Failed(실패) 비율을 보여주는 Pie Chart.
3. **`figures/performance_srcc.png`**: 최종 검토된 모델들 중 SRCC(상관계수) 지표를 가지는 모델들의 성능 비교 Bar Chart.
4. **전체 파이프라인 구조도**: `final_professor_report.md` 내부에 Mermaid 마크다운 포맷으로 작성.

### 생성하지 못한 시각자료

| 누락된 시각자료 | 이유 | 대안 |
| :--- | :--- | :--- |
| 점수 분포 그래프 | 본 주차별 보고서 맥락에서는 주차별 서사가 길어 단일 모델 분포도보단 타임라인/상태 위주 시각화가 적합하다고 판단함. | 필요 시 이전 최종 보고서의 분석 산출물 참고 가능 |
| 예측값 vs 정답 Scatter | 위와 동일 | - |
| Hard case 이미지 그리드 | 모델 학습 "기록"에 초점을 맞춘 주차별 감사에서는 각 주차별로 모든 이미지를 생성하는 것이 토큰 및 보고서 가독성에 불리함. | - |
| 모바일 적용성 비교표 (수치형) | 명시적인 Inference Time(ms), 파일 사이즈(MB) 데이터가 모든 모델에 대해 일관되게 기록된 CSV가 존재하지 않음. | 모델 상태표(`model_inventory.csv`)의 'TFLite/Flutter' 여부 컬럼으로 대체함. |
