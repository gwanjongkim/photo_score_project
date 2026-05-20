# TechIQA-Guard v1 Dataset Builder Context Notes

- Scope is dataset preparation only. No model training, TFLite export, Flutter edits, or stable trainer changes are in scope.
- The Gemini Stage 0 report exists at `outputs/tech_iqa_guard_v1_dataset_design_20260520/report.md`, but its stale `delta > 15`, forced `normalized_mos=0.2`, and `test_vila/` assumptions are intentionally not followed.
- The builder should use `delta_mixed112_existing >= 5` for hard false positives and `>= 8` for strong hard false positives.
- Manual user-flagged false positives are always included: `20230201_181300.jpg`, `1675342165226-13.jpg`, and `1675342165226-3.jpg`.
- Manual filename discovery must search under `data/` and `outputs/` and keep filename-only missing rows if nothing is found.
- Existing root `checklist.md` and `context-notes.md` contain unrelated local task history, so this task keeps notes in `data/processed/techiqa_guard/` to avoid modifying user-owned files.
- Default train/val inputs are the existing `data/processed/topiq_replacement/mixed_112_train.csv` and `mixed_112_val.csv`; held-out test inputs stay in separate FLIVE/KonIQ/SPAQ files.
- Existing eval prediction CSVs under `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/` are used to populate teacher/mixed score columns, mark delta flags on test rows, and count candidate deltas.
- Because no saved `topiq_mixed112_*` or `[TechnicalIqaCompare]` technical comparison log was found locally, `hard_false_positive.csv` should remain manual-only unless such a log/report is provided.
- Final default run wrote 16,384 train rows, 2,048 val rows, 3,981 FLIVE test rows, 1,008 KonIQ test rows, 1,125 SPAQ test rows, 3 manual hard false positives, and a 450-row smoke manifest.
- The three manual filenames were found under `outputs/test_vila_aesthetic_direct_compare_20260504_153044/thumbs/` with additional duplicate thumb copies in adjacent timestamped output folders.
