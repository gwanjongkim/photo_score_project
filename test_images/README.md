# Private Test Images

Put local scoring images in this folder before running the Aesthetic Weight Lab.

Supported default extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

Do not commit private images. This repository ignores image files in `test_images/` and keeps only this README plus `.gitkeep`.

Example:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  --input_dir test_images \
  --config configs/aesthetic_weight_lab.yaml \
  --output_dir outputs/aesthetic_weight_lab_demo
```
