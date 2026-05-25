# Mobile A-LAMP v2 Smoke Report

This is an A-LAMP-inspired and RGNet-inspired mobile branch, not an official A-LAMP reproduction.

## Command

```bash
PYTHONPATH=. /home/omen_pc1/photo_score_project/.venv_gpu/bin/python src/train/train_mobile_alamp_v2.py --train_patch_jsonl outputs/alamp_v4_full_ava_20260517/subsets/train_patch_boxes_full_v4.jsonl --val_patch_jsonl outputs/alamp_v4_full_ava_20260517/subsets/val_patch_boxes_full_v4.jsonl --out_dir outputs/mobile_alamp_v2_full_ava_20260519 --batch_size 4 --epochs 20 --save_model --backbone mobilenetv3small --backbone_weights imagenet --freeze_backbone --feature_dim 256 --attention_dim 128 --class_weight auto
```

## Inputs

- train patch JSONL: `outputs/alamp_v4_full_ava_20260517/subsets/train_patch_boxes_full_v4.jsonl`
- val patch JSONL: `outputs/alamp_v4_full_ava_20260517/subsets/val_patch_boxes_full_v4.jsonl`
- image size: `384`
- patch size/count: `224` / `5`
- preprocessing mode: `mobilenetv3_include_preprocessing_float_pixels_0_255`

## Dataset

```json
{
  "train_label_distribution": {
    "label_threshold": 5.0,
    "negative": 59689,
    "positive": 144717,
    "total": 204406
  },
  "train_requested_records": 204406,
  "val_label_distribution": {
    "label_threshold": 5.0,
    "negative": 7599,
    "positive": 17952,
    "total": 25551
  },
  "val_requested_records": 25551
}
```

## Model

```json
{
  "attention_dim": 128,
  "backbone": "mobilenetv3small",
  "backbone_weights": "imagenet",
  "feature_dim": 256,
  "freeze_backbone": true,
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
  "parameter_count": 2370785
}
```

## Class Weights

```json
{
  "applied": true,
  "mode": "auto",
  "weights": {
    "0": 1.7122585400995158,
    "1": 0.7062266354332939
  }
}
```

## Final Metrics

```json
{
  "accuracy": 0.910490095615387,
  "auc": 0.9746899604797363,
  "loss": 0.2008776217699051,
  "val_accuracy": 0.7141794562339783,
  "val_auc": 0.7459967136383057,
  "val_loss": 1.1200000047683716
}
```

## Prediction Diagnostics

```json
{
  "average_precision": 0.8640085179001559,
  "average_precision_error": null,
  "confusion_matrix_at_0_5": {
    "fn": 4320,
    "fp": 2983,
    "tn": 4616,
    "tp": 13632
  },
  "positive_prediction_ratio_at_0_5": 0.6502680912684435,
  "pred_max": 1.0,
  "pred_mean": 0.6491681933403015,
  "pred_min": 7.891953082891156e-26,
  "pred_std": 0.4070684313774109,
  "prediction_summary_json": "outputs/mobile_alamp_v2_full_ava_20260519/prediction_summary.json",
  "roc_auc": 0.7517757028788618,
  "sample_count": 25551,
  "val_predictions_csv": "outputs/mobile_alamp_v2_full_ava_20260519/val_predictions.csv"
}
```

## Notes

- Full 4096 training was not run by this smoke.
- TFLite export was not implemented in this step.
