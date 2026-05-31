# ICAA PTQ review plan

1. Inspect the existing PTQ conversion and smoke artifacts without modifying them.
2. Evaluate Dynamic PTQ against the FP16 reference on at least 50 deterministic images using the unchanged ICAA preprocessing contract.
3. Measure FP16 and Dynamic PTQ WSL interpreter latency as an experimental local-only signal.
4. Inspect the Full INT8 float-IO model, capture the invoke failure, and inventory DIV/op/tensor quantization details.
5. Write a conservative decision report that separates confirmed facts from assumptions.
