# Next Action

## Selected Path
C. Dynamic PTQ fails the A-cut parity threshold; move to QAT/student distillation.

## Rationale
- Dynamic PTQ has no NaNs or range errors, and it is faster in the WSL interpreter timing, but its output drift is too large for the stated A-cut gate.
- Overall MaxAE is `0.009905099868774414`, which is nearly 10x the `0.001` model-level threshold.
- `81 / 100` output values exceed `0.001`.
- Full INT8 float-IO is not usable because invoke fails consistently at quantized DIV node 593.

## Recommended Work
- Do not Flutter-test Dynamic PTQ yet.
- Do not create another deployment candidate from the current Full INT8 PTQ artifact.
- Prefer QAT or student distillation aimed at preserving ICAA output parity under the `0.001` threshold.
- If INT8 is revisited, start with a diagnostic-only experiment that keeps attention softmax/normalization DIV paths in float or otherwise avoids quantized denominator collapse, then rerun this same review.
