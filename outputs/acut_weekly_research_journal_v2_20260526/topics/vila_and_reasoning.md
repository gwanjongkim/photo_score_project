# VLM 및 시각적 이유 생성 연구

## 1. 확인된 사실

- **2026-03-30 ~ 2026-04-05**: Stage 5 및 Firebase 초기 통합
- **2026-04-06 ~ 2026-04-12**: Acut 제품 백엔드 정비 및 증류 학습
- **2026-04-13 ~ 2026-04-19**: VILA 및 NVILA-8B 멀티모달 추론 실험
- **2026-04-20 ~ 2026-04-26**: 경량 미적 품질 모델 확정 및 검증
- **2026-04-27 ~ 2026-05-03**: Gemma 어댑터 학습 및 모바일 수출 조사
- **2026-05-04 ~ 2026-05-10**: RGNet 및 A-LAMP 베이스라인 수립
- **2026-05-11 ~ 2026-05-17**: 그래프 기반 레이아웃 퓨전 및 ICAA TF 구현
- **2026-05-18 ~ 2026-05-24**: TOPIQ-lite 벤치마크 및 DistortionGuard 개발
- **2026-05-25 ~ 2026-05-31**: NIMA 버그 수정 및 최종 성능 재검증

## 2. 주요 근거

- `docs/acut_product_integration.md` (신뢰도: high)
- `outputs/acut_product_contract_check/top_k_summary.json` (신뢰도: high)
- `outputs/acut_product_contract_check_diversity/top_k_summary.json` (신뢰도: high)
- `outputs/acut_product_contract_check_smoke_diversity/top_k_summary.json` (신뢰도: high)
- `outputs/acut_product_firebase_regression/top_k_summary.json` (신뢰도: high)
- `outputs/firebase_worker_local_smoke/top_k_summary.json` (신뢰도: high)
- `outputs/firebase_worker_local_smoke_diversity/top_k_summary.json` (신뢰도: high)
- `outputs/acut_product_ready_smoke_diversity_post_integration/top_k_summary.json` (신뢰도: high)
- `outputs/acut_product_ready_smoke_post_integration/top_k_summary.json` (신뢰도: high)
- `outputs/acut_verify_now/top_k_summary.json` (신뢰도: high)

## 3. 현재 상태

VILA-lite 및 NVILA-8B를 이용한 이유 생성 파이프라인 구축됨.
