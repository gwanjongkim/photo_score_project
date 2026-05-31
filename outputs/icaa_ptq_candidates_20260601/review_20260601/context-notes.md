# ICAA PTQ review context notes

- Scope is diagnostic-only under `outputs/icaa_ptq_candidates_20260601/review_20260601/`.
- Production Flutter artifacts, production metadata, preprocessing contracts, and A-cut scoring formulas are out of scope and must remain untouched.
- Existing smoke evidence says Dynamic PTQ ran on 16 images with MAE `0.0030072899535298347`, MaxAE `0.008145809173583984`, no NaNs, and no range errors.
- Existing smoke evidence says Full INT8 float-IO conversion completed but invoke failed at TFLite `DIV` node 593 with a zero-denominator assertion.
- The current pass will treat the user-provided A-cut parity thresholds as evaluation gates, not as proof of production readiness.
- Extended Dynamic PTQ parity used 50 deterministic valid ICAA17K images after a discovered A-cut manifest had one unreadable image path.
- Dynamic PTQ failed the model-level `0.001` gate with overall MAE `0.003328498899936676`, MaxAE `0.009905099868774414`, P90 AE `0.006955149769783022`, and 81 of 100 output values over `0.001`.
- WSL timing showed FP16 mean `331.652110934956 ms` and Dynamic PTQ mean `191.28270630375482 ms`, but this is not a mobile benchmark.
- Full INT8 failed all four invoke probes at quantized DIV node 593; the denominator tensor had valid quantization parameters but contained runtime zero values.
