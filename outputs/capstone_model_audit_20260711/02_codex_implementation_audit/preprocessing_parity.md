# Preprocessing parity

| Model/path | Training/validation | Export/TFLite contract | Runtime risk | Status |
|---|---|---|---|---|
| KonIQ/FLIVE mobile | `csv_dataset.decode_img`: JPEG RGB → float `[0,1]`; train resize 256, random 224 crop/flip; eval direct resize 224 (`src/datasets/csv_dataset.py:7-19`) | Metadata: float32 NHWC RGB 224, divide by 255 | MobileNetV2 includes internal `Rescaling(1/127.5,-1)` in modern Keras, which expects `[0,255]`; feeding `[0,1]` compresses input near -1. This is a high-risk preprocessing mismatch unless the serialized version behaved differently. | MISMATCH / HIGH |
| NIMA fixed | RGB float `[0,1]`; train 256 resize/random crop/flip; eval center crop (`ava_distribution_dataset.py:45-60`) | Fixed model explicitly `include_preprocessing=False` (`nima_distribution.py:40-54`) | Coherent fixed source; old `nima_mobile.tflite` traces to legacy checkpoint, not fixed candidate. | MATCH fixed; MISMATCH deployment lineage |
| RGNet practical | Generic AADB input evidence is `[0,1]`, 256 square | TFLite input `[1,256,256,3]` float | EfficientNetV2B0 default includes preprocessing intended for `[0,255]`; source does not disable it. Duplicate/under-scaling risk parallels technical models. | MISMATCH risk / HIGH |
| A-LAMP practical | Global 384 plus five 224 patches; heuristic selector | TFLite inputs proven `[1,384,384,3]` and `[1,5,224,224,3]` | Backbones use default internal preprocessing; external range must match exact trainer/export. | UNVERIFIED |
| Mobile A-LAMP v2 | Export metadata explicitly says `mobilenetv3_include_preprocessing_float_pixels_0_255` | TFLite float32 two-input contract; FP16 weight quantization, float I/O | Bounded random test used `[0,1]` only to test execution, not correct scoring. Runtime must supply `[0,255]`. | MATCH within export artifact; mobile runtime UNVERIFIED |
| MUSIQ practical | Native/aspect-preserving scale loader, 32 patches, `[0,1]`; train flip | multi-input probe TFLites exist but were not executed in this batch | Fixed token count and masked-mean behavior differ from paper; runtime input assembly is nontrivial. | INTENTIONAL_ADAPTATION |
| MDIQA | Official configs use 384 crop/resize and ImageNet normalization in PyTorch path | no TFLite/ONNX found | No mobile parity path. | NOT_APPLICABLE mobile |

Validation/test augmentation is deterministic for generic regression (`training=False` direct resize) and NIMA (center crop), but the direct resize distorts aspect ratio for KonIQ/FLIVE. Paper-native or full-image behavior is therefore not preserved.

No integer-quantized core model was observed in the bounded set; FP16 artifacts retain float32 inputs/outputs. Quantized input dequantization parity is therefore NOT_APPLICABLE for those files.
