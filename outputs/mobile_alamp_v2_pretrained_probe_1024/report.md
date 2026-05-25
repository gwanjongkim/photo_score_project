# Mobile A-LAMP v2 Smoke Report

This is an A-LAMP-inspired and RGNet-inspired mobile branch, not an official A-LAMP reproduction.

## Command

```bash
PYTHONPATH=. /home/omen_pc1/photo_score_project/.venv_gpu/bin/python src/train/train_mobile_alamp_v2.py --out_dir outputs/mobile_alamp_v2_pretrained_probe_1024 --max_train_samples 1024 --max_val_samples 1024 --batch_size 8 --epochs 10 --save_model --backbone mobilenetv3small --backbone_weights imagenet --freeze_backbone --feature_dim 256 --attention_dim 128 --class_weight auto
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
    "negative": 295,
    "positive": 729,
    "total": 1024
  },
  "train_requested_records": 1024,
  "val_label_distribution": {
    "label_threshold": 5.0,
    "negative": 302,
    "positive": 722,
    "total": 1024
  },
  "val_requested_records": 1024
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
    "0": 1.735593220338983,
    "1": 0.7023319615912208
  }
}
```

## Final Metrics

```json
{
  "accuracy": 0.0,
  "auc": 0.0,
  "loss": 0.0,
  "val_accuracy": 0.7041015625,
  "val_auc": 0.6565279960632324,
  "val_loss": 0.6630388498306274
}
```

## Prediction Diagnostics

```json
{
  "average_precision": 0.7882993610878993,
  "average_precision_error": null,
  "confusion_matrix_at_0_5": {
    "fn": 124,
    "fp": 179,
    "tn": 123,
    "tp": 598
  },
  "positive_prediction_ratio_at_0_5": 0.7587890625,
  "pred_max": 1.0,
  "pred_mean": 0.6799204349517822,
  "pred_min": 0.04114830866456032,
  "pred_std": 0.24075362086296082,
  "prediction_summary_json": "outputs/mobile_alamp_v2_pretrained_probe_1024/prediction_summary.json",
  "roc_auc": 0.6565051090605566,
  "sample_count": 1024,
  "val_predictions_csv": "outputs/mobile_alamp_v2_pretrained_probe_1024/val_predictions.csv"
}
```

## Notes

- Full 4096 training was not run by this smoke.
- TFLite export was not implemented in this step.
