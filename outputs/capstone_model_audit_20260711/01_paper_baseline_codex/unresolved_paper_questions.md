# Unresolved paper questions

1. Obtain the AVA official paper/dataset page and determine whether any official filename split exists; the IEEE page was not opened.
2. NIMA’s rendered equations dropped the exact two learning-rate values and input dimensions. Verify them from the PDF source, and identify official backbone preprocessing/color order.
3. Locate any author-maintained NIMA, RGNet, A-LAMP, and AADB repositories or supplements. None was verified in this pass.
4. Resolve A-LAMP’s `{max,mean}` versus `{max,min}` internal inconsistency and exact loss/batch/schedule/checkpoint rule.
5. Open `ICAA17K_code/models/*`, `dataset.py`, and `main_nni.py` file contents to establish exact DAT return heads, label columns/scales, split, transform, loss, optimizer, batch, and model-selection criterion.
6. Establish whether ICAA17K confidence is only metadata/sample weighting or a supervised target. The paper does not support treating it as an output head.
7. Open the KonCept512 official repository linked by the dataset page; capture official split filenames, normalization, augmentation, optimizer, checkpoint selection, and whether PLCC is fitted.
8. Inspect PaQ-2-PiQ official training code for exact resize policy, ROI coordinates, pixel normalization, losses, optimizer/schedule and checkpoint criterion.
9. Retrieve the MDIQA PDF/full source successfully. Record exact technical/aesthetic head names and counts, label dimensions/scales, fusion equations, losses, input sampling, dataset splits, optimizer/scheduler, and every reported result table.
10. Open exact TOPIQ official config and architecture files. Record NR input/crop, normalization, loss, optimizer, epochs, split repetitions, test aggregation and logistic fitting.
11. Identify a primary publication, if any, explicitly claimed by `C6`, `RankCalib`, `TechIQA-Guard`, or `DistortionGuard`. Until then they have no paper requirements.
12. Identify the exact distillation paper/recipe claimed for Stage 5; “teacher-student” alone is insufficient.

These gaps are deliberately visible. No missing value was inferred from the local project or the interrupted Claude output.
