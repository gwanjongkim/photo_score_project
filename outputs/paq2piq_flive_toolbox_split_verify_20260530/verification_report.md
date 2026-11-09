# FLIVE / PaQ-2-PiQ Toolbox Split Verification

## 1. Summary
A. Verified: usable as IQA-Toolbox/PyIQA standardized split

## 2. Source
Metadata URL: https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets-metainfo/raw/main/meta_info_FLIVEDataset.csv
Source classification: toolbox-standardized metadata candidate; original-paper provenance not directly proven here.

## 3. Columns Found
- All columns: img_name/patch_name, mos, split, official_split
- Image column: img_name/patch_name
- Score column: mos
- Split column: official_split

## 4. Split Counts
- train: 121024
- val: 30908
- validation: 0
- test: 7308
- expected train/val/test: 30000 / 7700 / 1800
- image rows train/val/test: 30256 / 7727 / 1827
- patch rows train/val/test: 90768 / 23181 / 5481

## 5. Local File Matching
- total metadata rows: 159240
- matched local image count: 159240
- missing local image count: 0
- duplicate image ids: 0
- filename matching strategy: normalized path match against `data/raw/paq2piq`, `data/raw/flive`, plus filename fallback into common subdirectories (`images`, `flive`, `voc_emotic_ava`, `blur_dataset`, `patches`).

## 6. Score Validation
- image labels source checked: /home/omen_pc1/photo_score_project/data/raw/paq2piq/labels_image.csv
- patch labels source checked: /home/omen_pc1/photo_score_project/data/raw/paq2piq/labels_patch.csv
- matched rows against local labels: 159240
- exact score matches: 159240
- mean absolute score diff: 4.462087011806708e-20
- image mean absolute score diff: 0.0
- patch mean absolute score diff: 5.949449349075611e-20
- note: compared against local labels_image.csv and labels_patch.csv on normalized relative path keys.

## 7. Protocol Interpretation
- original paper exact reproduction: not proven by this metadata alone.
- MUSIQ/PyIQA-style research-standard reproduction: only if the split counts and filename match are both verified by this script.
- local custom fallback: do not use unless verification fails and the result is explicitly labeled custom.

## 8. Decision
A

## 9. Next Action
Recommend building final paper-protocol manifests using this metadata after manual provenance review.

## 10. Forbidden Actions
- Do not call local 80/10/10 split official.
- Do not train yet.
- Do not replace production technical_score.
- Do not use C6 as production.

## 11. Assumptions
- The same CSV may be consumable under either `official_split` or `split`; this script treats the first matching split-like column as authoritative for verification only.
- Local filename matching is approximate and intentionally conservative; a mismatch does not mutate data.

## 12. Verification Notes
- preview manifest written: yes
