# TOPIQ Teacher Validation Manifest Report

- input_csv: outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv
- output_csv: outputs/topiq_teacher_validation_20260531/validation_manifest.csv
- seed: 20260531
- input rows: 271095
- deduped image_path rows: 180438
- duplicate rows dropped: 90657

## Dataset Sampling

| dataset | available deduped | preferred val/test | train available | selected | train fill | selected splits |
|---|---:|---:|---:|---:|---:|---|
| koniq10k | 10073 | 8999 | 1074 | 300 | 0 | {'test': 300} |
| spaq | 11125 | 1125 | 10000 | 300 | 0 | {'test': 300} |
| flive | 159240 | 38216 | 121024 | 300 | 0 | {'test': 65, 'val': 235} |
