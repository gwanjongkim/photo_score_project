# Context Notes

- 2026-05-11 15:57 KST. Dependency gate passed in `./.venv_gpu`: `torch`, `torchvision`, and `ultralytics` were import-discoverable.
- The AVA CSVs use `image_path`, and current rows store repo-relative paths such as `data/raw/ava/images/165180.jpg`.
- This run is scoped to a 1024-image subset per split only. It is an A-LAMP-paper-oriented approximation and an object/global attribute graph approximation, not an official A-LAMP reproduction.
- The scripts are config-driven but keep CLI overrides for the requested fields. They update only aggregate `detection_summary.json` and `graph_summary.json` in the requested output folders.
- PyTorch is installed as `2.5.1+cu121`, but `torch.cuda.is_available()` returned `False` in `./.venv_gpu`. TensorFlow `2.20.0` also reported no visible GPU in this process.
- No local `yolo11n.pt` or `yolov8n.pt` was found. Ultralytics attempted to download both candidates from GitHub and failed in the sandbox; the elevated retry was rejected, so detection and graph generation were stopped.
