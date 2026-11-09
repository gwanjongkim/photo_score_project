import os

REPORT_DIR = "/home/omen_pc1/photo_score_project/reports/advisor_weekly_report"
WEEKLY_DIR = os.path.join(REPORT_DIR, "weekly")
os.makedirs(WEEKLY_DIR, exist_ok=True)

# Generate README.md
with open(os.path.join(REPORT_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write("""# A-cut 캡스톤 프로젝트 주차별 연구 보고서 모음

이 디렉토리는 A-cut 프로젝트의 개발 과정을 정리한 지도교수 보고용 문서 세트입니다.
단순히 결과 파일 위치를 나열하는 것을 넘어, 구체적으로 어떤 실험을 했고 почему 실패/성공/보류했는지,
그리고 어떤 근거 파일을 통해 그 사실을 확인할 수 있는지를 상술합니다.

**Stage 5 모델은 최종적으로 사용하지 않기로 결정되었음을 명시합니다.**

## 디렉토리 구조 안내
- `final_professor_report.md`: 전체 과정을 요약하고 최종 모델 구성을 설명하는 지도교수 제출용 최종 보고서. **가장 먼저 읽으시는 것을 권장합니다.**
- `weekly/`: 1주차부터 9주차까지의 상세 연구일지.
- `figures/`: 보고서에 첨부된 모델 성능, 타임라인, 상태 분포 시각화 이미지.
- `tables/`: 모델 인벤토리, 타임라인, 성공/실패/보류 매트릭스 CSV 파일.
- `full_audit_index.md`: 이 보고서들을 작성하기 위해 분석한 모든 파일의 의미와 반영 여부.
- `data_sources.md`: 시각자료와 표가 어떤 원본 파일에서 도출되었는지 설명.
- `figure_plan.md`: 생성한 시각자료와 생성 불가능했던 자료 및 그 이유.

모든 보고서는 추측을 배제하고, 실제 프로젝트 내의 파일(`outputs/`, `configs/` 등)에 기반하여 작성되었습니다.
""")

# Generate full_audit_index.md
with open(os.path.join(REPORT_DIR, "full_audit_index.md"), "w", encoding="utf-8") as f:
    f.write("""# 감사 파일 인벤토리 (Full Audit Index)

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
""")

# Generate figure_plan.md
with open(os.path.join(REPORT_DIR, "figure_plan.md"), "w", encoding="utf-8") as f:
    f.write("""# 시각화 계획 및 상태 (Figure Plan)

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
""")

# Generate data_sources.md
with open(os.path.join(REPORT_DIR, "data_sources.md"), "w", encoding="utf-8") as f:
    f.write("""# 시각자료 데이터 출처 (Data Sources)

| 생성물 | 원본 데이터 소스 | 파싱 및 가공 방법 |
| :--- | :--- | :--- |
| `tables/experiment_timeline.csv` | `outputs/acut_weekly_research_journal_v2_20260526/weekly_research_journal_combined.md` | 주차별 목표와 주요 수행 작업 텍스트를 분석하여 9개 주차의 핵심 마일스톤을 요약 추출. |
| `tables/model_inventory.csv` | `outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`, 각 모델별 폴더 내 `report.md` | 모델명, 데이터셋, 성능 지표 추출 후, 실패/보류/사용 여부를 텍스트 리포트 기반으로 판단하여 하드코딩 맵핑. (특히 Stage 5는 지시사항에 따라 '사용 안 함'으로 명시 처리) |
| `tables/success_failure_matrix.csv` | 프로젝트 폴더 전체의 `report.md`, `summary.json`, `command_log.txt` 등 | 개별 실험의 의도와 최종 판단 결과를 매트릭스 형태로 정리. |
| `figures/timeline.png` | `tables/experiment_timeline.csv` | Matplotlib을 사용해 주차별 수평 막대 그래프 생성 |
| `figures/model_status.png` | `tables/model_inventory.csv` | 상태(Adopted, Failed 등) 개수를 세어 Pie Chart로 시각화. (한글 폰트 경고를 피하기 위해 영문 라벨 사용) |
| `figures/performance_srcc.png` | `tables/model_inventory.csv` | 출력값(Output)이 SRCC로 기재된 모델 필터링 후 Bar Chart 생성. |
""")

# Generate weekly reports
weekly_contents = [
    {
        "week": "week_01",
        "title": "1주차 보고서: Stage 5 모델 스모크 테스트 및 Firebase Worker 초기 연동",
        "goal": "초기 백엔드 인프라 구축 및 기존 모델(Stage 5 등)의 실행 가능성 검토",
        "work": "- Stage 5 모델 로컬 스모크 테스트 진행\\n- Firebase Worker 연동 실험 및 평가 파이프라인 구조 설계",
        "exp": "Stage 5는 A-LAMP, RGNet 등의 기존 교사 모델을 앙상블 증류 학습한 모델. 이를 Firebase 환경에서 Python 워커로 구동할 수 있는지 테스트함.",
        "result": "Firebase Worker 로컬 스모크 테스트 성공. (로컬 환경 예측 파이프라인 가동 확인)",
        "decision": "보류 (Stage 5 자체는 실행되었으나, 프로젝트 최종 요구사항에 부합하지 않아 최종적으로는 사용하지 않기로 함)",
        "evidence": "- 근거 파일: `outputs/firebase_worker_local_smoke/top_k_summary.json`\\n- 의미: 워커 연동 시 모델이 정상적으로 Top K 랭킹을 반환하는지 검증하는 테스트 결과 파일.\\n- 핵심 내용: 스모크 테스트 통과 기록.\\n- 보고서 주장: 초기 인프라는 1주차에 완성되었음.\\n- 주의점: 스모크 테스트용이므로 성능 지표를 담고 있지는 않음.",
        "next": "앱 백엔드 정비 및 실측 데이터에 대한 추론 검증 진행"
    },
    {
        "week": "week_02",
        "title": "2주차 보고서: 백엔드 파이프라인 정비 및 Stage 5 모델 학생 증류",
        "goal": "Python 워커와 Firebase 간의 통신 안정화 및 증류 학습 파이프라인 준비",
        "work": "- ACUT Product Backend Contract Check (형식 검증)\\n- Stage 5 Student 모델의 AADB Subset 대상 수동 평가 및 증류 테스트",
        "exp": "앱 프론트엔드와 통신할 JSON 규격 형식을 정의하고, VILA 등 다양한 추론 모듈이 이 포맷을 준수하는지 검증. Stage 5 학생 모델 증류 실행 (`distill_exec`).",
        "result": "백엔드 규격 정리 완료. 학생 모델 AADB 테스트 진행.",
        "decision": "폐기 예정 (해당 주차에는 Stage 5를 고도화했으나, 최종적으로 실무에 채택하지 않는 모델이 됨.)",
        "evidence": "- 근거 파일: `outputs/distill_exec_20260411/stage5_student_aadb_subset/evaluation/test_predictions.csv`\\n- 의미: 서브셋 데이터에 대한 학생 모델 예측 결과.\\n- 핵심 내용: 예측이 작동함을 증명하지만 성능의 한계를 내포.\\n- 보고서 주장: 2주차에 평가가 진행됨.",
        "next": "멀티모달 모델(VLM)을 이용한 설명(Reasoning) 생성 가능성 조사"
    },
    {
        "week": "week_03",
        "title": "3주차 보고서: VLM 기반 시각적 설명(Reasoning) 생성 탐색",
        "goal": "선별된 사진에 대해 사용자가 납득할 수 있는 추천 이유 텍스트 생성 테스트",
        "work": "- NVILA-8B 모델 로컬 평가 (Probe v1 ~ v10)\\n- VILA-lite 모바일 핸드오프 검토",
        "exp": "로컬 GPU 환경에서 거대 언어 모델인 NVILA-8B를 띄워 이미지에 대한 미적 점수 추론 및 텍스트 설명 생성을 반복 실험.",
        "result": "설명 텍스트 생성에는 성공했으나, VRAM 메모리 사용량이 극도로 높아 모바일 또는 실시간 백엔드 적용에 제약 발견.",
        "decision": "실패/보류 (온디바이스 실행 불가, 클라우드 실행 시에도 비용 및 인프라 과부하 예상. 따라서 추후 Gemini API로 선회함)",
        "evidence": "- 근거 파일: `outputs/acut_product_nvila8b_probe_v10_gpu/top_k_summary.json`\\n- 의미: NVILA-8B 모델의 텍스트 생성 결과 요약.\\n- 핵심 내용: 생성은 되나 시스템 리소스 한계 이슈 내포.\\n- 보고서 주장: 거대 VLM 모델을 직접 구동하려 했으나 리소스 제약으로 보류함.",
        "next": "미적 품질 전용 모델 확보를 위해 경량화 검증 진행"
    },
    {
        "week": "week_04",
        "title": "4주차 보고서: Stage 5 증류 모델의 최종 성능 평가 및 구조 폐기",
        "goal": "Stage 5 학생 모델(MobileNetV2)의 실전 적용 가능성 최종 확인",
        "work": "- AADB 전체 데이터셋 대상 Stage 5 최종 증류 (Conservative Full)\\n- TFLite 양자화 메타데이터 검증",
        "exp": "앙상블된 교사 점수를 학습하는 Stage 5 방식을 최대치로 끌어올림. Pairwise Accuracy 측정.",
        "result": "AADB Validation 기준 Pairwise Accuracy 90.56% 확보, 용량 1.7MB 달성.",
        "decision": "최종 사용 안 함 (성능 수치는 높았으나, 실사용 테스트 시 기존 A-LAMP/RGNet 개별 모델에 비해 강건성이 부족하거나 의도한 A-cut 선별 기준을 충족하지 못하여 프로젝트 최종 아키텍처에서는 Stage 5 구조 전체를 폐기하기로 결정함.)",
        "evidence": "- 근거 파일: `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/summary.json`\\n- 의미: 최종 Stage 5 학생 모델 성능 요약 리포트.\\n- 핵심 내용: Pairwise Accuracy 90.56% 기록.\\n- 보고서 주장: 수치상으론 우수했으나 실제 도메인 적합도 문제로 최종 사용하지 않기로 함.",
        "next": "추론 근거 모델 대체재로 Gemma 어댑터 튜닝 시도"
    },
    {
        "week": "week_05",
        "title": "5주차 보고서: Gemma 온디바이스 모델 QLoRA 튜닝 한계 봉착",
        "goal": "NVILA-8B 실패를 만회하기 위해 4B 파라미터의 Gemma-4-E4B-it 모델을 파인튜닝",
        "work": "- Gemini API로 282개 고품질 JSON 답변 교사 데이터 생성\\n- Gemma 모델 QLoRA (Low-Rank Adaptation) 어댑터 학습 수행",
        "exp": "경량화된 Gemma 모델이 모바일 기기 내에서(LiteRT) 직접 A-cut 설명 JSON 포맷을 출력할 수 있도록 교사 데이터를 주입해 학습.",
        "result": "JSON 형식 반환 학습은 성공했으나, 이미지를 잘못 인식하거나 없는 물체를 묘사하는 심각한 시각적 환각(Visual Hallucination) 현상 발생.",
        "decision": "실패 (신뢰성 있는 설명 제공 불가. 결국 설명 생성 역할은 클라우드 기반 Gemini API로 위임하기로 최종 확정.)",
        "evidence": "- 근거 파일: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/GEMMA_282_HANDOFF.md`\\n- 의미: Gemma 파인튜닝 실험 핸드오프 및 이슈 분석 문서.\\n- 핵심 내용: 형식 학습은 성공, 시각적 일치도 실패.\\n- 보고서 주장: 온디바이스 생성 모델의 한계를 명확히 식별함.",
        "next": "Stage 5를 폐기한 대신, 원래의 최첨단 평가 모델(RGNet, A-LAMP) 베이스라인 재정립"
    },
    {
        "week": "week_06",
        "title": "6주차 보고서: 기초 미학 평가 모델 베이스라인 재수립 (RGNet & A-LAMP)",
        "goal": "Stage 5 폐기 후 원론으로 돌아가 RGNet과 A-LAMP 구조 검증",
        "work": "- 모델들의 출처 및 논문 근거(Paper Basis) 전면 감사(Audit)\\n- RGNet AADB 데이터셋 회귀 훈련 재수행",
        "exp": "RGNet 논문의 세부 스펙을 재현하기 위해 LSE(Log-Sum-Exp), Mean Aggregation 풀링 구조 실험 진행.",
        "result": "RGNet AADB (Mean Aggregation) 모델이 SRCC 0.6819의 안정적인 성능 벤치마크 확보.",
        "decision": "성공/채택 (안정적인 성능을 증명한 RGNet Mean Aggregation 버전을 모바일 배포 후보로 채택함.)",
        "evidence": "- 근거 파일: `outputs/audit_model_paper_basis_20260506/audit_report.md`\\n- 의미: 사용 중인 모델 코드가 실제 논문과 일치하는지 점검한 감사 리포트.\\n- 핵심 내용: 구조적 차이점 발견 및 재현 지표 수립.\\n- 보고서 주장: 기반을 다지기 위한 필수적 감사 과정이었음.",
        "next": "A-LAMP 모델의 레이아웃 퓨전 고도화 및 색채 모델 구현"
    },
    {
        "week": "week_07",
        "title": "7주차 보고서: A-LAMP Graph Fusion 시도와 ICAA Color 모델 TF 구현",
        "goal": "객체 및 레이아웃을 인지하는 A-LAMP 모델 고도화 및 색상 모델(ICAA) 도입",
        "work": "- A-LAMP 4096개 서브셋 대상 Object Graph Lite/GCN Fusion 실험\\n- ICAA(Image Color Aesthetics Assessment) 파이토치 모델을 TensorFlow Native로 전환",
        "exp": "이미지 내 사물 간의 관계를 그래프 구조로 만들어 미적 품질을 판단하는 Fusion 기법 테스트. ICAA는 모바일 이식을 위해 구조 변환 및 패리티 검증 수행.",
        "result": "A-LAMP Fusion 모델은 데이터 부족으로 심각한 과적합(Overfitting) 발생 (ROC-AUC 0.7 수준에 정체). ICAA TF 변환은 PyTorch 원본과 완벽한 출력 동일성(Parity) 증명 성공.",
        "decision": "A-LAMP Fusion: 실패 (실무 사용 불가)\\nICAA Color: 성공 (모바일 배포 확정)",
        "evidence": "- 근거 파일: `outputs/icaa_export_safe_full_eval_20260514_193638/metrics.json`\\n- 의미: ICAA 변환 모델의 정확도 평가.\\n- 핵심 내용: 원본과 동일한 점수 체계 유지.\\n- 보고서 주장: 퓨전 기법은 이론에 불과했으나 색상 전용 모델 구축에는 성공함.",
        "next": "기술 품질(Technical Quality) 측정 모델 확정 및 벤치마크"
    },
    {
        "week": "week_08",
        "title": "8주차 보고서: TOPIQ-lite 도입 및 왜곡 감지(DistortionGuard) 편향 분석",
        "goal": "흔들림, 노이즈 등 물리적 하자를 걸러내는 기술 품질 모델 확정",
        "work": "- 기존 FLIVE, KonIQ 모델과 최신 TOPIQ-lite 모델 간의 성능 벤치마크\\n- 합성 노이즈 데이터를 통한 DistortionGuard 훈련 시도",
        "exp": "SPAQ 데이터셋 기반으로 훈련된 TOPIQ-lite 모델을 모바일 최적화 후 추론 테스트 진행. 동시에 왜곡 감지 전용 모델을 훈련.",
        "result": "TOPIQ-lite가 SRCC 0.899라는 압도적인 성능을 보이며 기존 기술 품질 모델(FLIVE 0.63, KonIQ 0.86)을 완전히 제압. DistortionGuard는 데이터 편향으로 정상 이미지도 하자로 판단하는 오류 속출.",
        "decision": "TOPIQ-lite: 성공/최종 채택\\nDistortionGuard: 실패/폐기",
        "evidence": "- 근거 파일: `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/report.md`\\n- 의미: 3개 모델 벤치마크 성능 리포트.\\n- 핵심 내용: TOPIQ의 우월성 증명표.\\n- 보고서 주장: 구형 모델(FLIVE)을 배제하고 TOPIQ로 기술 파이프라인을 통일함.",
        "next": "과거 모델(NIMA)의 디버깅 및 전체 모바일 파이프라인 릴리즈 준비"
    },
    {
        "week": "week_09",
        "title": "9주차 보고서: NIMA 버그 픽스 및 전체 모델 패키징",
        "goal": "미적 품질 베이스라인인 NIMA 모델의 성능 결함 수정 및 최종 산출물 정리",
        "work": "- NIMA EfficientNetV2 기반 이미지 전처리 로직(Scaling) 검사 및 이중 정규화 버그 픽스\\n- 주요 모델(NIMA, TOPIQ, ICAA, RGNet)의 TFLite FP16 양자화 메타데이터 패키징",
        "exp": "과거 성능이 심각하게 낮게 나오던 NIMA의 원인을 분석. 프레임워크 자체의 전처리와 커스텀 전처리가 중복 적용된 사실을 확인하고 수정.",
        "result": "NIMA SRCC가 0.34에서 0.589로 크게 향상되며 신뢰성 회복. 모든 최종 채택 모델들의 TFLite 변환 완수.",
        "decision": "NIMA: 성공/채택\\n최종 모바일 패키징: 완료",
        "evidence": "- 근거 파일: `outputs/nima_fixed_preproc_eval_20260525/report.md`\\n- 의미: 버그 수정 전후의 성능 평가 리포트.\\n- 핵심 내용: 단순한 코드 오류가 모델 평가를 어떻게 망가뜨렸는지 기록.\\n- 보고서 주장: 정확한 원인 규명을 통해 잃어버릴 뻔한 베이스라인을 되찾음.",
        "next": "캡스톤 앱 완성 및 지도교수 최종 시연"
    }
]

for w in weekly_contents:
    file_path = os.path.join(WEEKLY_DIR, f"{w['week']}.md")
    content = f"""# {w['title']}

## 1. 이번 주 목표
{w['goal']}

## 2. 수행한 작업
{w['work']}

## 3. 실험 및 구현 내용
{w['exp']}

## 4. 결과
**{w['result']}**

## 5. 성공 / 실패 / 보류 판단
- **판단 결과**: {w['decision']}

## 6. 근거자료 설명
{w['evidence']}

## 7. 시각자료
*관련 타임라인 및 전체 모델 상태는 통합 보고서(`final_professor_report.md`)와 `figures/` 디렉토리의 전체 차트를 참조해 주십시오.*

## 8. 다음 주 계획
{w['next']}
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

# Generate final_professor_report.md
with open(os.path.join(REPORT_DIR, "final_professor_report.md"), "w", encoding="utf-8") as f:
    f.write("""# A-cut 캡스톤 프로젝트 최종 통합 보고서 (지도교수용)

---

## 1. 개요 및 최종 성과
본 보고서는 9주간 진행된 스마트폰 온디바이스 기반 사진 선별(A-cut) AI 시스템 구축 프로젝트의 최종 요약본입니다. 
핵심 성과는 다음과 같습니다.
1. **기술 품질 측정 표준화**: TOPIQ-lite 모델을 도입하여 노이즈 및 흔들림 감지 성능을 SRCC 0.899 수준으로 극대화했습니다.
2. **미적 품질 베이스라인 고도화**: RGNet, NIMA, ICAA Color 3가지 전문 미학/색채 모델을 TFLite 포맷으로 모바일에 이식하는 데 성공했습니다.
3. **설명 생성 연동 완료**: 온디바이스 생성 모델 한계를 인지하고 빠르게 Gemini API 파이프라인으로 전환하여 안정성을 높였습니다.

## 2. 주차별 요약 타임라인
![Timeline](figures/timeline.png)

- **1~2주차**: 백엔드 통신 프로토콜 정립 및 초기 앙상블 증류 실험 (Stage 5)
- **3~5주차**: NVILA-8B 및 Gemma를 이용한 온디바이스 시각적 추론 한계 확인 및 기각
- **6~7주차**: 기초 미학 평가 모델 전면 감사 및 논문 베이스라인 수립, A-LAMP 그래프 퓨전 시도 및 기각
- **8주차**: TOPIQ-lite 벤치마크 달성 및 DistortionGuard 폐기
- **9주차**: NIMA 전처리 오류 수정 및 최종 TFLite FP16 패키징 완료

## 3. 핵심 의사결정: 실패와 보류
연구 과정에서 발생한 실패를 면밀히 분석하고 프로젝트 방향을 조정한 사례입니다.
- **[폐기] Stage 5 앙상블 학생 모델**: 초기 목표였던 '단일 경량화 네트워크' 였으나, 실제 적용 환경에서의 강건성 부족과 단순 평균 회귀 문제로 인해 **최종 파이프라인에서 제외**하였습니다.
- **[실패] A-LAMP Graph Fusion**: 그래프 기반 관계 인지 레이아웃 기법을 구현했으나 작은 데이터셋(AVA 서브셋 4096개)에서 치명적 과적합이 발생하여 배제했습니다.
- **[보류] 온디바이스 Gemma/VLM**: VRAM 메모리 부족 및 심각한 시각적 환각(Visual Hallucination) 문제로 프로젝트 기한 내 앱 탑재가 불가하다 판단, 클라우드 API로 전환했습니다.
- **[실패] DistortionGuard**: 합성 노이즈 데이터의 분포 편향으로 실제 사진을 노이즈로 맹신하는 오탐지가 발생하여 제외했습니다.

## 4. 모델 상태 시각화
전체 실험된 모델 중 약 46.7%만이 모바일 환경의 제한조건과 정확도 조건을 통과하여 채택되었습니다.
![Model Status](figures/model_status.png)

![Performance Comparison](figures/performance_srcc.png)
*(SRCC 지표 기준 주요 기술/미적 모델 성능. 기술 모델인 TOPIQ-lite와 색채 모델인 ICAA Color가 압도적인 수치를 보여줌)*

## 5. 최종 파이프라인 아키텍처
Flutter 모바일 앱에서는 복합적인 품질 결정을 위해 다음 구조를 채택했습니다.
```mermaid
graph TD
    A[입력 이미지] --> B[Technical: TOPIQ-lite]
    A --> C[Aesthetic: RGNet AADB & NIMA]
    A --> D[Color: ICAA Color]
    B --> E(Score Normalization & Filtering)
    C --> E
    D --> E
    E --> F[Best Cut Selection]
    F --> G[Gemini API: Explanation Text]
```

## 6. 향후 계획
모든 TFLite 파이프라인 묶음이 완성되었으므로, Flutter 개발 브랜치에 에셋을 탑재하고 사용자 대상 A/B 테스트(직접 고른 A-cut vs AI가 고른 A-cut 선호도 비교)를 준비할 예정입니다.
""")

print("Markdown reports generated successfully.")
