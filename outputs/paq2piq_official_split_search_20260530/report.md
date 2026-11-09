# PaQ-2-PiQ Official Split Search

## 1. Summary
C. Not found; only custom split exists.

Confirmed facts:
- Raw PaQ-2-PiQ/FLIVE labels exist locally with the expected totals: 39,810 image labels and 119,430 patch labels.
- No local raw label file inspected contains an official `train`/`validation`/`test` partition column.
- The local processed image splits are 31,848 / 3,981 / 3,981, which is an 80/10/10 split and does not match the paper target of about 30,000 / 7,700 / 1,800 pictures.
- The local processed patch splits are 95,544 / 11,943 / 11,943, which is an 80/10/10 split and does not match the paper target of about 90,000 / 23,100 / 5,400 patches.
- Local split-generation scripts use custom 80/10/10 logic or default rebuilt patches to `train`; none reconstruct the paper protocol.

Assumptions:
- Approximate paper counts were evaluated against the user-provided protocol numbers only. I did not infer alternative official counts.
- Files that are derived prediction outputs, training logs, model exports, or unrelated aesthetic datasets are not treated as split candidates unless they contain PaQ-2-PiQ/FLIVE split evidence.

## 2. Search Scope
Directories searched:
- `data/raw/paq2piq/`
- `data/processed/paq2piq/`
- `data/raw/flive/`
- `data/processed/flive/`
- `scripts/`
- `src/`
- `external/`
- `outputs/`
- `configs/`

Additional local evidence checked because related FLIVE artifacts appeared during search:
- `data/processed/techiqa_guard/test_flive.csv`
- `data/processed/topiq_replacement/flive_pairs_gap005_10k.csv`
- `data/processed/topiq_replacement/flive_pairs_gap010_10k.csv`
- `data/processed/smoke/flive_1024_train.csv`
- `data/processed/smoke/flive_1024_val.csv`

## 3. Candidate Files
| Path | Type | Why relevant | Row count | Split labels | Paper count match |
|---|---|---|---:|---|---|
| `data/raw/paq2piq/labels_image.csv` | csv | PaQ/FLIVE image MOS labels | 39,810 | none | no; total image labels match paper total but no split labels |
| `data/raw/paq2piq/labels_patch.csv` | csv | PaQ/FLIVE patch MOS labels | 119,430 | none | no; total patch labels match paper total but no split labels |
| `data/raw/paq2piq/patches_rebuilt.csv` | csv | Rebuilt patch manifest has a split column | 119,430 | train | no |
| `data/raw/flive/labels_image.csv` | csv | FLIVE image MOS labels | 39,810 | none | no; total image labels match paper total but no split labels |
| `data/raw/flive/labels_patch.csv` | csv | FLIVE patch MOS labels | 119,430 | none | no; total patch labels match paper total but no split labels |
| `data/raw/flive/.ipynb_checkpoints/labels_image-checkpoint.csv` | csv | Notebook checkpoint label duplicate | 39,810 | none | no |
| `data/processed/paq2piq/image_train.csv / image_val.csv / image_test.csv` | csv group | Local image split | 39,810 | train/val/test | no |
| `data/processed/paq2piq/patch_train.csv / patch_val.csv / patch_test.csv` | csv group | Local patch split | 119,430 | train/val/test | no |
| `data/processed/paq2piq/all_train.csv / all_val.csv / all_test.csv` | csv group | Combined image+patch split | 159,240 | train/val/test | no |
| `data/processed/flive/image_train.csv / image_val.csv / image_test.csv` | csv group | Local FLIVE image split | 39,810 | train/val/test | no |
| `data/processed/techiqa_guard/test_flive.csv` | csv | Downstream test derivative | 3,981 | test | no |
| `data/processed/topiq_replacement/flive_pairs_gap005_10k.csv` | csv | Derived FLIVE pair data | 10,000 | none | no |
| `data/processed/topiq_replacement/flive_pairs_gap010_10k.csv` | csv | Derived FLIVE pair data | 10,000 | none | no |
| `data/processed/smoke/flive_1024_train.csv / flive_1024_val.csv` | csv group | Smoke subset | 1,152 | train/val | no |
| `src/data/build_flive_csv.py` | python | Split-generation script | n/a | train/val/test | no |
| `src/preprocess/make_flive_csv.py` | python | PaQ-2-PiQ split-generation script | n/a | train/val/test | no |
| `src/preprocess/make_spaq_flive_csv.py` | python | SPAQ/FLIVE manifest builder | n/a | train/val/test | no |
| `src/data/prepare_flive.py` | python | Patch reconstruction script | n/a | train | no |
| `src/data/flive_utils.py` | python | Patch helper module checked for split logic | n/a | none | no |
| `outputs/paper_protocol_technical_iqa_audit_20260530/report.md` | markdown | Prior audit evidence | n/a | none | no |
| `outputs/paper_protocol_technical_iqa_audit_20260530/next_action.md` | markdown | Prior audit next step | n/a | none | no |

Details are also written to `candidate_split_files.csv`.

No candidate matched the paper split counts approximately. The only files with local train/val/test partitioning are custom/generated split files.

## 4. Existing Custom Splits
Local custom splits found:
- `data/processed/paq2piq/image_train.csv`, `image_val.csv`, `image_test.csv`: 31,848 / 3,981 / 3,981 image rows.
- `data/processed/paq2piq/patch_train.csv`, `patch_val.csv`, `patch_test.csv`: 95,544 / 11,943 / 11,943 patch rows.
- `data/processed/paq2piq/all_train.csv`, `all_val.csv`, `all_test.csv`: 127,392 / 15,924 / 15,924 mixed image+patch rows.
- `data/processed/flive/image_train.csv`, `image_val.csv`, `image_test.csv`: 31,848 / 3,981 / 3,981 image rows.
- `data/processed/techiqa_guard/test_flive.csv`: 3,981 rows, a downstream custom test derivative.
- `data/processed/topiq_replacement/flive_pairs_gap005_10k.csv` and `flive_pairs_gap010_10k.csv`: derived pairwise ranking data, not paper split manifests.
- `data/processed/smoke/flive_1024_train.csv` and `flive_1024_val.csv`: smoke subsets only.

These are not official PaQ-2-PiQ paper protocol splits unless separately proven by an external official source. The local code evidence points the opposite way: `src/data/build_flive_csv.py` and `src/preprocess/make_flive_csv.py` use `train_test_split` with `random_state=42`; `src/preprocess/make_spaq_flive_csv.py` shuffles and slices 80/10/10; `src/data/prepare_flive.py` writes every rebuilt patch row as `split=train` because patch split information was absent.

## 5. Paper Protocol Compatibility
PaQ-2-PiQ paper protocol cannot be reproduced locally from the currently found files.

What is reproducible locally:
- The raw image-label total of 39,810 pictures.
- The raw patch-label total of 119,430 patches.
- Custom FLIVE/PaQ-2-PiQ 80/10/10 image and patch splits.

What is missing:
- Official mapping of images into about 30,000 train, 7,700 validation, and 1,800 test pictures.
- Official mapping of patches into about 90,000 train, 23,100 validation, and 5,400 test patches.
- Evidence that the local test set is the official test set with at least one image dimension greater than 640.

## 6. Next Action
C. Search/download official split externally.

Use the official PaQ-2-PiQ/FLIVE repository, author-provided data release, paper supplementary files, or a cited benchmark implementation to recover the original split IDs. After that, build paper-protocol manifests from the recovered split only after manual verification of source provenance and counts.

## 7. Forbidden Actions
- Do not create custom split silently.
- Do not call custom split paper protocol.
- Do not train yet.
- Do not replace production technical_score.
