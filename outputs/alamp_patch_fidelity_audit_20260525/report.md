# A-LAMP Patch Fidelity Audit

## 1. Summary
본 감사는 A-LAMP Multi-Patch teacher 모델의 성능 정체 원인을 진단하기 위해 수행되었습니다. 주요 가설은 패치 선택 알고리즘이나 전처리 과정이 원본 논문 및 외부 코드와 일치하지 않을 가능성이었습니다. 감사 결과, 패치 선택(MR Saliency + Diversity) 및 전처리(BGR mean subtraction, 224x224 crop)는 외부 코드와 높은 수준으로 일치함을 확인했습니다. 그러나 모델 아키텍처, 특히 **Aggregation Head의 크기**와 **Patch Feature 추출 방식**에서 원본 논문과 상당한 차이가 발견되었습니다.

## 2. External A-LAMP Patch Pipeline
- **Saliency Map:** Manifold Ranking (MR) 알고리즘을 사용합니다.
- **Patch Selection:** Saliency, Pattern Diversity (Earth Mover's Distance), Euclidean distance를 조합한 목적 함수를 Nelder-Mead 방식으로 최적화하여 5개의 패치를 선택합니다.
- **Preprocessing:** `cv2.imread`(BGR) -> Crop -> BGR Mean Subtraction (`[103.939, 116.779, 123.68]`). 별도의 Resize 없이 224x224 크기를 직접 추출합니다.
- **Backbone:** VGG16 (include_top=False) 뒤에 2개의 FC layer (4096 nodes)를 추가하여 각 패치에서 4096차원 특징을 추출합니다. (Flatten -> FC4096 -> FC4096)
- **Aggregation:** 5개 패치의 특징을 Mean, Max로 각각 통합한 후 Concatenate하여 8192차원 벡터를 만듭니다. 이후 2개의 FC layer (4096 nodes)를 거쳐 최종 점수를 예측합니다.

## 3. Current Project Patch Pipeline
- **Saliency/Selection:** 외부에서 생성된 JSONL 패치 박스를 그대로 사용하며, 이는 외부 코드로 생성된 것이 확인되었습니다.
- **Preprocessing:** `PIL.Image`(RGB) -> `vgg16.preprocess_input` (Caffe mode) -> Resize(224, 224, BILINEAR). Keras의 전처리는 내부적으로 RGB를 BGR로 변환하고 동일한 Mean 값을 빼주므로 외부 코드와 수학적으로 동일합니다.
- **Backbone:** VGG16 (include_top=False) + **GlobalAveragePooling2D (GAP)**를 사용하여 각 패치에서 512차원 특징을 추출합니다.
- **Aggregation:** Mean + Max Concatenate (1024차원) -> **Dense(256)** -> Dense(1).

## 4. Paper vs External vs Current Comparison
| Feature | A-LAMP Paper / External | Current Project | Status |
| :--- | :--- | :--- | :--- |
| Patch Size | 224x224 | 224x224 | Match |
| Preprocessing | BGR Subtraction | RGB -> BGR Subtraction | Match |
| Patch Feature | **4096-dim (Flatten+FC)** | **512-dim (GAP)** | **Mismatch (Low Capacity)** |
| Aggregation Head | **Mean+Max -> FC(4096)x2** | **Mean+Max -> FC(256)** | **Mismatch (Very Low Capacity)** |
| Unfreeze Block | Block 5 unfreeze | Block 5 unfreeze | Match |
| Saliency | MR Saliency | MR Saliency (via JSONL) | Match |

## 5. Coordinate and Coverage Statistics
- **Avg Patch Area Ratio:** ~0.166 (전형적인 AVA 이미지 크기 대비 적절)
- **Max IoU (Overlap):** 평균 0.450. 패치 간의 중첩이 존재하지만 과도하지 않음.
- **Out of Bounds:** 0.5% 미만으로 무시 가능한 수준.
- **Near Duplicates (IoU > 0.9):** 1.3% 미만. 패치 다양성이 잘 유지되고 있음.
- **Center Distribution:** 중심에서 평균 0.29 거리(정규화 좌표)에 위치하여, 너무 중앙에만 쏠리지 않고 분산되어 있음.

## 6. Patch Statistics for Error Cases
- **False Negatives (FN):** `max_iou`가 0.508로, 정상 예측(0.443)보다 높게 나타남. 패치 간의 중첩이 많아 시각적 정보가 제한될 때 FN이 발생할 가능성이 시사됨.
- **False Positives (FP):** `avg_iou`가 0.113으로 소폭 높지만, 다른 지표는 정상 예측과 유사함.
- **Conclusion:** 패치 통계 자체는 에러 케이스에서도 극단적인 이상을 보이지 않음. 패치 품질보다는 모델의 해석 능력(Capacity) 문제일 가능성이 큼.

## 7. Likely Causes of Performance Ceiling
1.  **Aggregation Head Capacity (Critical):** 논문은 4096 노드 2계층을 사용하지만, 현재는 256 노드 1계층만 사용합니다. 5개 패치의 복잡한 상호작용을 학습하기에 부족할 수 있습니다.
2.  **Patch Feature Loss:** 논문은 Flatten+FC를 통해 공간 정보를 보존하며 고차원 특징을 뽑지만, 현재는 GAP를 통해 512차원으로 압축합니다. 이 과정에서 세부적인 구도 정보가 소실될 수 있습니다.
3.  **Missing Layout-Aware Subnet:** A-LAMP의 성능은 전체 이미지를 보는 Layout-aware subnet과의 결합에서 나옵니다. 현재는 Multi-patch subnet만 사용하고 있어 근본적인 성능 한계가 존재합니다.

## 8. Recommended Next Step
- **Action A (High Priority):** Aggregation Head를 논문 규격(FC 4096 x 2) 또는 적어도 현재보다 크게(예: FC 1024 x 2) 확장하십시오.
- **Action B (Medium Priority):** GAP 대신 Flatten + Dense(4096)를 사용하여 패치 특징의 차원을 높이십시오.
- **Action C:** 현재 패치 박스와 전처리는 신뢰할 수 있으므로, 패치 재생성보다는 모델 구조 개선에 집중하십시오.
- **Action D:** Class-weight는 Negative Recall 개선에는 효과적이나 Calibration을 해칠 수 있으므로, Head 확장 후 다시 검토하십시오.

## 9. Final Judgment
패치 정합성(Fidelity)은 양호하며, 현재의 성능 정체는 **모델 레이어의 용량(Capacity) 부족** 및 **Aggregation 계층의 단순함**에서 기인한 것으로 판단됩니다. 패치 전처리를 수정할 필요는 없으며, 모델 아키텍처를 원본 논문에 더 가깝게 보강하는 것을 권장합니다.
