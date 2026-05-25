# AVA Pretrain + AADB Fine-tune Checklist

- [x] Create dated experiment output directory.
- [x] Run and log environment checks.
- [x] Verify AADB `score` target scale for train/val.
- [x] Inspect RGNet and A-LAMP training scripts for pretrained-load support.
- [x] Create experiment-only fine-tune scripts if existing scripts cannot load pretrained models.
- [x] Fine-tune RGNet from AVA checkpoint on AADB.
- [x] Fine-tune A-LAMP from AVA checkpoint on AADB.
- [x] Evaluate baseline, AVA-only, and fine-tuned RGNet on fixed AVA/AADB val512 subsets.
- [x] Evaluate baseline, AVA-only, and fine-tuned A-LAMP on fixed AVA/AADB val512 subsets.
- [x] Export fine-tuned RGNet and A-LAMP to TFLite under the experiment output directory.
- [x] Verify TFLite parity for both fine-tuned models.
- [x] Write `metrics_summary.json`.
- [x] Write `finetune_report.md`.
