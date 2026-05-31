# ICAA QAT attempt plan

1. Audit feasibility from the existing SavedModel, source TensorFlow model, PyTorch checkpoint, and installed Python tooling.
2. If direct QAT is feasible, create a short conservative teacher-distillation QAT workflow under this directory.
3. If direct QAT is blocked, do not fake a QAT run; document the blocker, write final reports, and stop ICAA quantization attempts for now.
4. Keep all generated scripts, reports, checkpoints, and candidate models under `outputs/icaa_qat_attempt_20260601/`.
