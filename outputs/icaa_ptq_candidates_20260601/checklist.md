# ICAA PTQ candidate checklist

- [x] Confirm existing SavedModel, FP16 TFLite reference, and ICAA17K images exist.
- [x] Keep production Flutter artifacts, production metadata, preprocessing contract, and A-cut formulas untouched.
- [x] Copy Stage K converter to Stage L.
- [x] Implement dynamic-range PTQ conversion with float32 input/output.
- [x] Implement full-integer INT8 PTQ conversion with representative ICAA17K data and float32 input/output if supported.
- [x] Write `conversion_report.md` and `model_inventory.csv`.
- [x] Generate `smoke_compare_ptq.py`.
- [x] Run Stage L conversion.
- [x] Run PTQ smoke comparison if candidates are produced.
- [x] Review artifact inventory and report final metrics.
