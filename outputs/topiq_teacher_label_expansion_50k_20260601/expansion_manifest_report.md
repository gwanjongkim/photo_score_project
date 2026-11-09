# TOPIQ 50k Label Expansion Manifest Report

- input_csv: outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv
- validation_manifest_csv: outputs/topiq_teacher_validation_20260531/validation_manifest.csv
- holdout_manifest_csv: outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest.csv
- dryrun_manifest_csv: outputs/topiq_teacher_label_dryrun_20260601/dryrun_manifest.csv
- exclude_dryrun: False
- output_csv: outputs/topiq_teacher_label_expansion_50k_20260601/expansion_manifest.csv
- seed: 20260601
- input rows: 271095
- deduped rows before exclusion: 180438
- duplicate rows dropped: 90657
- validation image paths excluded: 900
- holdout image paths excluded: 900
- dry-run image paths considered for exclusion: 0
- excluded union image paths: 1800
- raw input rows excluded by selected exclusion paths: 7200
- deduped rows after exclusion: 178638
- selected rows: 50000

## Dataset Sampling

| dataset | available after exclusion | requested | selected | took all available | unique source groups | multi-row source groups | split distribution | content type distribution |
|---|---:|---:|---:|---|---:|---:|---|---|
| koniq10k | 9473 | 9000 | 9000 | False | 9000 | 0 | {'test': 1813, 'train': 7187} | {'image': 9000} |
| spaq | 10525 | 9000 | 9000 | False | 9000 | 0 | {'test': 446, 'train': 8554} | {'image': 9000} |
| flive | 158640 | 32000 | 32000 | False | 23549 | 7258 | {'test': 1508, 'train': 24372, 'val': 6120} | {'image': 7896, 'patch': 24104} |
