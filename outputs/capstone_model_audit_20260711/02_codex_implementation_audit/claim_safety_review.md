# Claim safety review

## Safe with qualification

- “NIMA-style 10-bin distribution model trained with EMD” is supported for the fixed source.
- “RGNet-inspired model with explicit region graph reasoning” is supported. “RGNet reproduction” is not.
- “Mobile A-LAMP-inspired five-patch attention model” is supported. “Faithful A-LAMP” is not.
- “KonIQ-trained MobileNetV2 technical regressor” and “FLIVE-image-trained MobileNetV2 regressor” are supported.
- “Local practical MUSIQ-style transformer” is supported; “official MUSIQ” is unsafe.
- “Official MDIQA code/checkpoint benchmark” can be used where checkpoint hash and official config are named; “locally trained faithful MDIQA” is unproven.

## Unsafe or unsupported

- Calling KonIQ mobile “KonCept512” or FLIVE image mobile “PaQ-2-PiQ.”
- Claiming the old NIMA export contains the preprocessing-fixed checkpoint.
- Claiming ICAA head semantics, mobile latency or deployed exclusion without direct port/runtime evidence.
- Calling ensemble correlations independently verified when the inspected prediction table lacks ground truth.
- Presenting technical blend components as commensurate without explicit 0–100/0–1/1–10 conversion.
- Claiming Stage 5/MDIQA students were deployed merely because TFLite or Keras files exist.
- Reporting rejected/no-go conclusions as paper-fidelity findings.

Recommended naming uses `-trained`, `-inspired`, `practical`, or `mobile adaptation` explicitly.
