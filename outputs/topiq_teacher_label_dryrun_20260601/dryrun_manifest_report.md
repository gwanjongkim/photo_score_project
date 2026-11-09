# TOPIQ Label Dry-Run Manifest Report

- input_csv: outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv
- validation_manifest_csv: outputs/topiq_teacher_validation_20260531/validation_manifest.csv
- holdout_manifest_csv: outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest.csv
- output_csv: outputs/topiq_teacher_label_dryrun_20260601/dryrun_manifest.csv
- seed: 20260601
- input rows: 271095
- deduped rows before exclusion: 180438
- duplicate rows dropped: 90657
- validation image paths excluded: 900
- holdout image paths excluded: 900
- excluded union image paths: 1800
- raw input rows excluded by validation/holdout paths: 7200
- deduped rows after exclusion: 178638

## Dataset Sampling

| dataset | available after exclusion | requested | selected | took all available | split distribution | content type distribution |
|---|---:|---:|---:|---|---|---|
| koniq10k | 9473 | 3000 | 3000 | False | {'test': 607, 'train': 2393} | {'image': 3000} |
| spaq | 10525 | 3000 | 3000 | False | {'test': 153, 'train': 2847} | {'image': 3000} |
| flive | 158640 | 3000 | 3000 | False | {'test': 122, 'train': 2281, 'val': 597} | {'image': 765, 'patch': 2235} |
