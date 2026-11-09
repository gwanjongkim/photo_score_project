# 5주차 보고서: Gemma 온디바이스 모델 QLoRA 튜닝 한계 봉착

## 1. 이번 주 목표
NVILA-8B 실패를 만회하기 위해 4B 파라미터의 Gemma-4-E4B-it 모델을 파인튜닝

## 2. 수행한 작업
- Gemini API로 282개 고품질 JSON 답변 교사 데이터 생성\n- Gemma 모델 QLoRA (Low-Rank Adaptation) 어댑터 학습 수행

## 3. 실험 및 구현 내용
경량화된 Gemma 모델이 모바일 기기 내에서(LiteRT) 직접 A-cut 설명 JSON 포맷을 출력할 수 있도록 교사 데이터를 주입해 학습.

## 4. 결과
**JSON 형식 반환 학습은 성공했으나, 이미지를 잘못 인식하거나 없는 물체를 묘사하는 심각한 시각적 환각(Visual Hallucination) 현상 발생.**

## 5. 성공 / 실패 / 보류 판단
- **판단 결과**: 실패 (신뢰성 있는 설명 제공 불가. 결국 설명 생성 역할은 클라우드 기반 Gemini API로 위임하기로 최종 확정.)

## 6. 근거자료 설명
- 근거 파일: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/GEMMA_282_HANDOFF.md`\n- 의미: Gemma 파인튜닝 실험 핸드오프 및 이슈 분석 문서.\n- 핵심 내용: 형식 학습은 성공, 시각적 일치도 실패.\n- 보고서 주장: 온디바이스 생성 모델의 한계를 명확히 식별함.

## 7. 시각자료
*관련 타임라인 및 전체 모델 상태는 통합 보고서(`final_professor_report.md`)와 `figures/` 디렉토리의 전체 차트를 참조해 주십시오.*

## 8. 다음 주 계획
Stage 5를 폐기한 대신, 원래의 최첨단 평가 모델(RGNet, A-LAMP) 베이스라인 재정립
