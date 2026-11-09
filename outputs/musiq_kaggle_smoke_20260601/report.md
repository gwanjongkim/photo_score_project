# Official Kaggle MUSIQ Smoke Test

## Scope
- Smoke test only.
- No training, TFLite export, Flutter edits, production scoring edits, or distillation.
- Local repo-practical MUSIQ was not imported or used.

## Packages
- tensorflow: 2.21.0
- tensorflow-hub: 0.16.1
- kagglehub: 1.0.1
- setuptools: 80.10.2 (pinned below 81 so tensorflow-hub can import pkg_resources)
- kagglehub fallback was not needed when tensorflow_hub direct loading succeeded

## Images
- /home/omen_pc1/photo_score_project/test_samples/KakaoTalk_20260330_180646779.jpg
- /home/omen_pc1/photo_score_project/test_samples/KakaoTalk_20260330_180646779_01.jpg
- /home/omen_pc1/photo_score_project/test_samples/KakaoTalk_20260330_180646779_02.jpg
- /home/omen_pc1/photo_score_project/test_samples/KakaoTalk_20260330_180646779_03.jpg
- /home/omen_pc1/photo_score_project/test_samples/KakaoTalk_20260330_180646779_04.jpg

## Variant Status
### koniq-10k
- handle: `https://www.kaggle.com/models/google/musiq/TensorFlow2/koniq-10k/1`
- load_method: `tensorflow_hub_direct`
- loaded: `True`
- output_scale_hint: `likely 0..100`
### spaq
- handle: `https://www.kaggle.com/models/google/musiq/TensorFlow2/spaq/1`
- load_method: `tensorflow_hub_direct`
- loaded: `True`
- output_scale_hint: `likely 0..100`
### paq2piq
- handle: `https://www.kaggle.com/models/google/musiq/TensorFlow2/paq2piq/1`
- load_method: `tensorflow_hub_direct`
- loaded: `True`
- output_scale_hint: `likely 0..100`

## Raw Outputs
- koniq-10k | KakaoTalk_20260330_180646779.jpg | signature_image_bytes_tensor_bytes_scalar | [50.370460510253906]
- koniq-10k | KakaoTalk_20260330_180646779_01.jpg | signature_image_bytes_tensor_bytes_scalar | [46.02167892456055]
- koniq-10k | KakaoTalk_20260330_180646779_02.jpg | signature_image_bytes_tensor_bytes_scalar | [48.73417282104492]
- koniq-10k | KakaoTalk_20260330_180646779_03.jpg | signature_image_bytes_tensor_bytes_scalar | [41.04152297973633]
- koniq-10k | KakaoTalk_20260330_180646779_04.jpg | signature_image_bytes_tensor_bytes_scalar | [42.61762619018555]
- spaq | KakaoTalk_20260330_180646779.jpg | signature_image_bytes_tensor_bytes_scalar | [46.096343994140625]
- spaq | KakaoTalk_20260330_180646779_01.jpg | signature_image_bytes_tensor_bytes_scalar | [49.39756393432617]
- spaq | KakaoTalk_20260330_180646779_02.jpg | signature_image_bytes_tensor_bytes_scalar | [53.724143981933594]
- spaq | KakaoTalk_20260330_180646779_03.jpg | signature_image_bytes_tensor_bytes_scalar | [22.95502471923828]
- spaq | KakaoTalk_20260330_180646779_04.jpg | signature_image_bytes_tensor_bytes_scalar | [27.756412506103516]
- paq2piq | KakaoTalk_20260330_180646779.jpg | signature_image_bytes_tensor_bytes_scalar | [68.4849624633789]
- paq2piq | KakaoTalk_20260330_180646779_01.jpg | signature_image_bytes_tensor_bytes_scalar | [62.10719299316406]
- paq2piq | KakaoTalk_20260330_180646779_02.jpg | signature_image_bytes_tensor_bytes_scalar | [67.71418762207031]
- paq2piq | KakaoTalk_20260330_180646779_03.jpg | signature_image_bytes_tensor_bytes_scalar | [59.44013595581055]
- paq2piq | KakaoTalk_20260330_180646779_04.jpg | signature_image_bytes_tensor_bytes_scalar | [62.70600509643555]

## Final Decision
MUSIQ_READY_FOR_100_IMAGE_SANITY
