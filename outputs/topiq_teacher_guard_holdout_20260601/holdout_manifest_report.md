# TOPIQ Guarded Holdout Manifest Report

- input_csv: outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv
- validation_manifest_csv: outputs/topiq_teacher_validation_20260531/validation_manifest.csv
- output_csv: outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest.csv
- seed: 20260601
- input rows: 271095
- deduped rows: 180438
- duplicate rows dropped: 90657
- validation rows excluded: 900
- raw input rows excluded by validation paths: 3600
- deduped rows after exclusion: 179538

## Dataset Sampling

| dataset | available deduped after exclusion | preferred val/test | train available | selected | train fill | selected splits |
|---|---:|---:|---:|---:|---:|---|
| koniq10k | 9773 | 8699 | 1074 | 300 | 0 | {'test': 300} |
| spaq | 10825 | 825 | 10000 | 300 | 0 | {'test': 300} |
| flive | 158940 | 37916 | 121024 | 300 | 0 | {'test': 59, 'val': 241} |
