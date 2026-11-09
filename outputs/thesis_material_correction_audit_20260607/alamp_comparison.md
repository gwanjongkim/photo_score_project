# 오리지널 A-LAMP vs A-cut 구현체 (Mobile A-LAMP v2) 비교 분석

이 문서는 오리지널 A-LAMP(논문 기준)와 본 레포지토리에서 실제 TFLite 배포용으로 재설계된 `mobile_alamp_v2` (`src/models/mobile_alamp_v2.py`) 간의 차이점을 비교합니다. 

이 비교는 논문에서 "모바일 탑재를 위해 불가피하게 포기한 연산들과 그에 따른 구조 개선(Engineering Trade-offs)"을 설명할 때 매우 중요한 증거 자료가 됩니다.

| 구성 요소 (Component) | Original A-LAMP (논문) | A-cut 구현체 (`mobile_alamp_v2`) | 변경 사유 (Reason for Change) | 모바일 탑재에 미치는 영향 (Effect on Mobile Feasibility) | 코드 증거 경로 (Evidence Path) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **백본 네트워크 (Backbone)** | VGG 등 무거운 모델 기반 다중 컬럼(Multi-column) | `MobileNetV3Small` | 모델 크기 및 연산량 대폭 감소, 모바일(TFLite)에서 실시간 추론 확보 | 파라미터가 획기적으로 줄어 모바일 메모리 한계 통과 | `src/models/mobile_alamp_v2.py` 내 `_build_mobilenetv3small` 함수 |
| **패치 추출 (Patch Extraction)** | Saliency map, Pattern diversity 등 동적(Dynamic) 연산을 통한 적응형 패치 추출 | 외부에서 미리 전처리된 크기의 고정 패치(예: `patch_count=5`)를 Input Tensor로 직접 받음 | Saliency map 계산과 동적 NMS/오버랩 제한 로직은 TFLite 환경에서 호환성이 나쁘고 오버헤드가 극심함 | 텐서 크기를 고정시켜 TFLite 정적 그래프 컴파일이 가능해짐 | `train_mobile_alamp_v2.py` (전처리된 패치 로드), `mobile_alamp_v2.py`의 `patches` Input |
| **속성 그래프 (Attribute Graph)** | 패치 간 관계 및 레이아웃을 모델링하기 위해 GCN 등의 그래프 신경망(Graph Neural Network) 사용 | **제거됨**. 대신 단순화된 Scaled Dot-Product Attention (Self-Attention)으로 토큰 간 관계 요약 | 그래프 연산과 Object Detection 노드 생성 로직의 과도한 복잡성 및 TFLite 변환 버그 회피 | GCN 대비 연산이 극히 단순화되어 모바일 GPU/NPU 가속기에서 병목 없이 동작 가능 | `mobile_alamp_v2.py` 내 `_region_layout_branch` 함수 (GCN 대신 기본 Attention 사용) |
| **레이아웃 인식 (Layout-aware Subnet)** | 객체 탐지(Object Detection) 결과 바운딩 박스를 활용한 정교한 공간 배치 파악 | 패치 외의 전체 이미지(`full_image`) 브랜치에서 추출된 공간 특징에 1x1 Conv 및 Attention을 걸어 전역 맥락(Global Context)으로 대용 | 별도의 무거운 Object Detector를 로드할 필요성 제거 | 단일 MobileNetV3 통과만으로 Layout 특징을 근사하여 극도의 연산 이득 | `mobile_alamp_v2.py` 내 `layout_backbone` 구성 및 `Concatenate` 로직 |
| **융합 및 최종 점수 (Fusion)** | 복잡한 계층적 융합 및 다중 랭킹 손실 적용 | Patch branch 평균 특징 벡터 + Layout branch 평균 특징 벡터 단순 `Concatenate` 후 Dense 레이어 통과 | TFLite 지원 기본 연산자(Built-in Ops)만으로 추론 그래프 구성 위함 (`SELECT_TF_OPS` 방지) | 별도의 커스텀 옵레이터 없이 100% 모바일 네이티브 지원 | `mobile_alamp_v2.py` 내 `fused = tf.keras.layers.Concatenate...` 부분 |

## 논문에서의 활용 전략
본 수정 사항들을 논문에 기록할 때, "성능을 유지하면서 단순화했다"고 과장하기보다는 **"객체 탐지와 그래프 구조가 미적 평가 성능엔 기여하지만 모바일 온디바이스 연산엔 치명적인 병목이 되므로, 이를 경량화된 Self-attention 및 사전 추출 고정 패치 기반 MobileNetV3 듀얼 브랜치로 근사(Approximate)하여 실용적인 Trade-off를 달성했다"**라고 서술하는 것이 매우 설득력 있습니다.