# 8주차 보고서: TOPIQ-lite 도입 및 왜곡 감지(DistortionGuard) 편향 분석

## 1. 이번 주 목표
흔들림, 노이즈 등 물리적 하자를 걸러내는 기술 품질 모델 확정

## 2. 수행한 작업
- 기존 FLIVE, KonIQ 모델과 최신 TOPIQ-lite 모델 간의 성능 벤치마크\n- 합성 노이즈 데이터를 통한 DistortionGuard 훈련 시도

## 3. 실험 및 구현 내용
SPAQ 데이터셋 기반으로 훈련된 TOPIQ-lite 모델을 모바일 최적화 후 추론 테스트 진행. 동시에 왜곡 감지 전용 모델을 훈련.

## 4. 결과
**TOPIQ-lite가 SRCC 0.899라는 압도적인 성능을 보이며 기존 기술 품질 모델(FLIVE 0.63, KonIQ 0.86)을 완전히 제압. DistortionGuard는 데이터 편향으로 정상 이미지도 하자로 판단하는 오류 속출.**

## 5. 성공 / 실패 / 보류 판단
- **판단 결과**: TOPIQ-lite: 성공/최종 채택\nDistortionGuard: 실패/폐기

## 6. 근거자료 설명
- 근거 파일: `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/report.md`\n- 의미: 3개 모델 벤치마크 성능 리포트.\n- 핵심 내용: TOPIQ의 우월성 증명표.\n- 보고서 주장: 구형 모델(FLIVE)을 배제하고 TOPIQ로 기술 파이프라인을 통일함.

## 7. 시각자료
*관련 타임라인 및 전체 모델 상태는 통합 보고서(`final_professor_report.md`)와 `figures/` 디렉토리의 전체 차트를 참조해 주십시오.*

## 8. 다음 주 계획
과거 모델(NIMA)의 디버깅 및 전체 모바일 파이프라인 릴리즈 준비
