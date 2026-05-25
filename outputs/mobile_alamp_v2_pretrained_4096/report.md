# Mobile A-LAMP v2 Smoke Report

This is an A-LAMP-inspired and RGNet-inspired mobile branch, not an official A-LAMP reproduction.

## Command

```bash
PYTHONPATH=. /home/omen_pc1/photo_score_project/.venv_gpu/bin/python src/train/train_mobile_alamp_v2.py --train_patch_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/train_patch_boxes_4096_v4.jsonl --val_patch_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_4096_v4.jsonl --out_dir outputs/mobile_alamp_v2_pretrained_4096 --max_train_samples 4096 --max_val_samples 4096 --batch_size 4 --epochs 20 --save_model --backbone mobilenetv3small --backbone_weights imagenet --freeze_backbone --feature_dim 256 --attention_dim 128 --class_weight auto
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
    "negative": 1174,
    "positive": 2922,
    "total": 4096
  },
  "train_requested_records": 4096,
  "val_label_distribution": {
    "label_threshold": 5.0,
    "negative": 1196,
    "positive": 2900,
    "total": 4096
  },
  "val_requested_records": 4096
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
    "0": 1.7444633730834753,
    "1": 0.7008898015058179
  }
}
```

## Final Metrics

```json
{
  "accuracy": 0.0,
  "auc": 0.0,
  "loss": 0.0,
  "val_accuracy": 0.720947265625,
  "val_auc": 0.6919431090354919,
  "val_loss": 1.2428309917449951
}
```

## Prediction Diagnostics

```json
{
  "average_precision": 0.8354220833823471,
  "average_precision_error": null,
  "confusion_matrix_at_0_5": {
    "fn": 387,
    "fp": 756,
    "tn": 440,
    "tp": 2513
  },
  "positive_prediction_ratio_at_0_5": 0.798095703125,
  "pred_max": 1.0,
  "pred_mean": 0.785893440246582,
  "pred_min": 1.5511355221098366e-12,
  "pred_std": 0.343505859375,
  "prediction_summary_json": "outputs/mobile_alamp_v2_pretrained_4096/prediction_summary.json",
  "roc_auc": 0.7028870660823434,
  "sample_count": 4096,
  "val_predictions_csv": "outputs/mobile_alamp_v2_pretrained_4096/val_predictions.csv"
}
```

## Notes

- Full 4096 training was not run by this smoke.
- TFLite export was not implemented in this step.
