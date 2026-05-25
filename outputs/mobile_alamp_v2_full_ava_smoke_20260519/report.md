# Mobile A-LAMP v2 Smoke Report

This is an A-LAMP-inspired and RGNet-inspired mobile branch, not an official A-LAMP reproduction.

## Command

```bash
PYTHONPATH=. /home/omen_pc1/photo_score_project/.venv_gpu/bin/python src/train/train_mobile_alamp_v2.py --train_patch_jsonl outputs/alamp_v4_full_ava_20260517/subsets/train_patch_boxes_full_v4.jsonl --val_patch_jsonl outputs/alamp_v4_full_ava_20260517/subsets/val_patch_boxes_full_v4.jsonl --out_dir outputs/mobile_alamp_v2_full_ava_smoke_20260519 --max_train_samples 256 --max_val_samples 128 --batch_size 4 --epochs 1 --smoke --save_model --backbone mobilenetv3small --backbone_weights imagenet --freeze_backbone --feature_dim 256 --attention_dim 128 --class_weight auto
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
    "negative": 67,
    "positive": 189,
    "total": 256
  },
  "train_requested_records": 256,
  "val_label_distribution": {
    "label_threshold": 5.0,
    "negative": 36,
    "positive": 92,
    "total": 128
  },
  "val_requested_records": 128
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
    "0": 1.9104477611940298,
    "1": 0.6772486772486772
  }
}
```

## Final Metrics

```json
{
  "accuracy": 0.51171875,
  "auc": 0.43883758783340454,
  "loss": 0.785468578338623,
  "val_accuracy": 0.734375,
  "val_auc": 0.672705352306366,
  "val_loss": 0.6193386912345886
}
```

## Prediction Diagnostics

```json
{
  "average_precision": 0.8375490211882766,
  "average_precision_error": null,
  "confusion_matrix_at_0_5": {
    "fn": 9,
    "fp": 25,
    "tn": 11,
    "tp": 83
  },
  "positive_prediction_ratio_at_0_5": 0.84375,
  "pred_max": 0.7367119789123535,
  "pred_mean": 0.5629383325576782,
  "pred_min": 0.37269043922424316,
  "pred_std": 0.06593231856822968,
  "prediction_summary_json": "outputs/mobile_alamp_v2_full_ava_smoke_20260519/prediction_summary.json",
  "roc_auc": 0.6739130434782609,
  "sample_count": 128,
  "val_predictions_csv": "outputs/mobile_alamp_v2_full_ava_smoke_20260519/val_predictions.csv"
}
```

## Notes

- Full 4096 training was not run by this smoke.
- TFLite export was not implemented in this step.
