# Checklist

- [x] Confirm `torch`, `torchvision`, and `ultralytics` are available in `./.venv_gpu`.
- [x] Inspect AVA CSV image path schema.
- [x] Create isolated object extraction and graph generation scripts.
- [x] Run `py_compile` with `./.venv_gpu/bin/python`.
- [ ] Run 1024-image detection for train, val, and test only. Blocked: no local YOLO weights and network download approval was rejected.
- [ ] Generate fixed-size graph JSONL files for train, val, and test. Blocked until detection JSONL exists.
- [ ] Validate JSONL parsing and fixed-size graph fields. Blocked until graph JSONL exists; graph shape smoke test passed on a synthetic in-memory record.
- [x] Write report, summary JSON, and command log.
