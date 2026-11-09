# ICAA 제외 사유 및 논문 활용 노트 (ICAA Exclusion Note)

## 1. 성능적 성공 (Performance Success)
ICAA Color 모델은 미적 모델들 중 단일 모델 기준 매우 뛰어난 색채 미적 평가 성능(ICAA17K 테스트 기준 SRCC 0.8791)을 입증했습니다.

## 2. 변환 및 정합성의 성과 (Conversion & Parity Success)
해당 모델은 원래 PyTorch 구조였으나, 모바일(TFLite)로 올리기 위한 ONNX 변환 과정에서 치명적인 버그가 발생했습니다. 프로젝트에서는 이를 포기하지 않고 TensorFlow Native API로 모델 로직을 완전히 재구현(`experiments/icaa_tf_native/`)하는 훌륭한 엔지니어링 투자를 단행하여 안드로이드-WSL 간 완벽한 TFLite 변환과 정합성(Parity)을 확보했습니다.

## 3. 모바일 지연 속도 한계 (Mobile Latency Failure)
변환과 성능 확보라는 성과에도 불구하고, 최종 안드로이드 기기 상에서 구동했을 때 **목표 추론 속도(Latency)를 만족하지 못하는 치명적 병목**이 발견되었습니다. 다른 경량 앙상블(NIMA, RGNet, Mobile A-LAMP) 대비 자원 소모가 너무 커 실시간 베스트 샷 선택 환경을 저해했습니다.

## 4. 최종 배제 사유 (Final Exclusion Reason)
이에 따라, 프로젝트 오너는 모바일 유저 경험(UX) 확보를 위해 성능이 우수한 ICAA를 최종 기본 모바일 앙상블(Default Production Ensemble)에서 의도적으로 배제하는 용단을 내렸습니다.

## 5. 논문에서의 활용 방안 (Thesis Usage)
논문에서는 "우리가 좋은 모델 4개를 다 썼다"고 포장하면 안 됩니다.
대신 **"모바일 환경에서의 트레이드오프 한계점 (Limitations and Edge Device Trade-offs)"** 챕터에 단독 섹션을 할당하여 이 사례를 서술해야 합니다.
> "본 연구에서는 성능이 우수한 ICAA 구조를 Native TF로 직접 포팅하여 모바일 변환 제약을 돌파하는 성과를 거두었다. 그러나 최종 디바이스에서의 실제 Latency 측정 결과, 모바일 배포 스펙을 초과함이 확인되었다. 이는 모바일 딥러닝 시스템에서 정량적 성능(SRCC)과 변환 가능성이 확보되더라도, 최종 애플리케이션 요구사항(속도)에 맞지 않으면 프로덕션에서 배제해야 함을 시사한다."