# Python-to-TFLite parity

## Independently verified this phase

No same-input Python-versus-TFLite pair was independently executed. The bounded tests verified TFLite execution only. Status: **UNVERIFIED** for independent parity.

## Raw pre-existing parity artifact inspected

Mobile A-LAMP v2’s `parity_report.json` contains per-sample Keras, FP32 and FP16 predictions for 1,024 validation samples and an exact export command. It is not accepted solely as a report: its paths, model IO and TFLite files exist, and the FP16 file was independently loaded/invoked.

| Variant | Input contract | Max abs diff | Mean abs diff | Correlation | Quantization |
|---|---|---:|---:|---|---|
| FP32 | full image `[N,384,384,3]`, patches `[N,5,224,224,3]`, float `[0,255]` | .0111042 | .00136761 | Pearson .999983, Spearman .999973 | float32 |
| FP16 | same float32 I/O | .0276954 | .00267344 | Pearson .999895, Spearman .999865 | FP16 weights, float32 I/O |

The report does not provide a separate “final score difference” beyond the scalar output difference; for this scalar classifier they are the same. The bounded FP16 inference returned finite scalar `.06437` on seeded but incorrectly scaled `[0,1]` input, so that value is not a quality score.

KonIQ, FLIVE, legacy NIMA, practical RGNet/A-LAMP and Stage5 student have verification JSON leads, but same-input raw arrays were not independently replayed here. They remain **UNVERIFIED** for parity despite passing TFLite inference.
