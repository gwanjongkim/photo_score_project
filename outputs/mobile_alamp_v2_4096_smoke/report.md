# Mobile A-LAMP v2 Smoke Report

This is an A-LAMP-inspired and RGNet-inspired mobile branch, not an official A-LAMP reproduction.

## Command

```bash
PYTHONPATH=. /home/omen_pc1/photo_score_project/.venv_gpu/bin/python src/train/train_mobile_alamp_v2.py --train_patch_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/train_patch_boxes_4096_v4.jsonl --val_patch_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_4096_v4.jsonl --out_dir outputs/mobile_alamp_v2_4096_smoke --max_train_samples 128 --max_val_samples 64 --batch_size 2 --epochs 1 --smoke --save_model
```

## Inputs

- train patch JSONL: `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/train_patch_boxes_4096_v4.jsonl`
- val patch JSONL: `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_4096_v4.jsonl`
- image size: `384`
- patch size/count: `224` / `5`

## Dataset

```json
{
  "train_label_distribution": {
    "label_threshold": 5.0,
    "negative": 33,
    "positive": 95,
    "total": 128
  },
  "train_requested_records": 128,
  "val_label_distribution": {
    "label_threshold": 5.0,
    "negative": 20,
    "positive": 44,
    "total": 64
  },
  "val_requested_records": 64
}
```

## Model

```json
{
  "input_shapes": {
    "full_image": [
      null,
      384,
      384,
      3
    ],
    "patches": [
      null,
      5,
      224,
      224,
      3
    ]
  },
  "name": "mobile_alamp_v2",
  "output_shape": [
    null,
    1
  ],
  "parameter_count": 692785
}
```

## Final Metrics

```json
{
  "accuracy": 0.6875,
  "auc": 0.5387560129165649,
  "loss": 0.6547465920448303,
  "val_accuracy": 0.6875,
  "val_auc": 0.45511364936828613,
  "val_loss": 0.6550196409225464
}
```

## Notes

- Full 4096 training was not run by this smoke.
- TFLite export was not implemented in this step.
