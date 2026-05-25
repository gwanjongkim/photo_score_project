# Mobile A-LAMP v2 Smoke Report

This is an A-LAMP-inspired and RGNet-inspired mobile branch, not an official A-LAMP reproduction.

## Command

```bash
PYTHONPATH=. /home/omen_pc1/photo_score_project/.venv_gpu/bin/python src/train/train_mobile_alamp_v2.py --out_dir outputs/mobile_alamp_v2_pretrained_smoke --max_train_samples 128 --max_val_samples 64 --batch_size 2 --epochs 1 --smoke --save_model --backbone mobilenetv3small --backbone_weights imagenet --freeze_backbone --feature_dim 256 --attention_dim 128 --class_weight auto
```

## Inputs

- train patch JSONL: `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/train_patch_boxes_4096_v4.jsonl`
- val patch JSONL: `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_4096_v4.jsonl`
- image size: `384`
- patch size/count: `224` / `5`
- preprocessing mode: `mobilenetv3_include_preprocessing_float_pixels_0_255`

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
    "0": 1.9393939393939394,
    "1": 0.6736842105263158
  }
}
```

## Final Metrics

```json
{
  "accuracy": 0.5078125,
  "auc": 0.4440191090106964,
  "loss": 0.7943532466888428,
  "val_accuracy": 0.65625,
  "val_auc": 0.7255681753158569,
  "val_loss": 0.6307371854782104
}
```

## Prediction Diagnostics

```json
{
  "average_precision": 0.8663784843487742,
  "average_precision_error": null,
  "confusion_matrix_at_0_5": {
    "fn": 8,
    "fp": 14,
    "tn": 6,
    "tp": 36
  },
  "positive_prediction_ratio_at_0_5": 0.78125,
  "pred_max": 0.7051477432250977,
  "pred_mean": 0.5481282472610474,
  "pred_min": 0.36978447437286377,
  "pred_std": 0.06769236922264099,
  "prediction_summary_json": "outputs/mobile_alamp_v2_pretrained_smoke/prediction_summary.json",
  "roc_auc": 0.7227272727272728,
  "sample_count": 64,
  "val_predictions_csv": "outputs/mobile_alamp_v2_pretrained_smoke/val_predictions.csv"
}
```

## Notes

- Full 4096 training was not run by this smoke.
- TFLite export was not implemented in this step.
