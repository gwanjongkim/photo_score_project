# A-LAMP External Repo / Weights Audit

## 1. Environment
- **Target Repository:** `https://github.com/GuillaumeBalezo/A-Lamp`
- **Framework:** TensorFlow 2 (Keras)
- **Local Path:** `external/A-Lamp_external_audit/A-Lamp/`

## 2. External Repository Summary
The repository is an unofficial implementation of the A-LAMP paper, created by members of the PRIM project. It contains a mix of Python scripts and Jupyter notebooks. While it implements the Multi-Patch Subnet and a Salient Object Detection (SOD) pipeline, the Layout-Aware Subnet file (`layout_aware_subnet.py`) is 0 bytes and completely missing. 

## 3. Official Status
1. **Is GuillaumeBalezo/A-Lamp official or unofficial?** Unofficial.
2. **Is there any evidence of an official author repository?** No. Web searches and references in subsequent papers (like TANet) state that official code is "N/A" (Not Available).
3. **Are official weights available anywhere?** No.
4. **Is there an author-provided AVA split/protocol?** No.

## 4. Framework and Dependencies
5. **What framework does it use?** TensorFlow 2 (Keras).
6. **Is it notebook-only or script-based?** A mix. It uses scripts for multi-patch training and notebooks for EDA and SOD.
7. **Does it include runnable training/inference code?** Yes, but only for the Multi-Patch Subnet (`train_complete.py`).
8. **What dependencies does it require?** TensorFlow 2, OpenCV, SciPy, scikit-image, NumPy, Pandas.

## 5. Implemented A-LAMP Components
9. **Does it implement Multi-Patch Subnet?** Yes.
10. **Does it implement Adaptive Patch Selection?** Yes, using SLIC superpixels and Graph-Based Manifold Ranking.
11. **Does it implement Orderless Aggregation?** Yes (concatenation of Average and Maximum pooling).
12. **Does it implement Layout-Aware Subnet?** No. The `layout_aware_subnet.py` file is 0 bytes.
13. **Does it implement Object/Global Attribute Graph?** No. 

## 6. Weights Availability
17. **Does the model weights link still exist?** The README contains a Google Drive link, but its contents cannot be verified as official.
18. **What file type and approximate size are the weights?** Expected to be `.hdf5` based on `train_complete.py`.
19. **Are the weights for Multi-Patch only or full A-LAMP?** Multi-Patch only.
20. **Are the weights usable without retraining?** Only for evaluating the Multi-Patch branch.
21. **Are license/usage terms visible?** The repo is MIT licensed, but the saliency detection algorithm explicitly forbids commercial use.

## 7. Saliency/Object/Graph Availability
14. **Does it include salient object detection?** Yes, it includes a TF2 port of "Unconstrained Salient Object Detection via Proposal Subset Optimization".
15. **Does it include object boxes, labels, or graph generation?** It outputs bounding boxes for salient objects, but no semantic labels or graph generation logic.
16. **Does it use AVA labels as mean_score > 5 or >= 5?** It uses boolean labels derived from the AVA dataset, but the exact threshold isn't clear without inspecting the dataset generation.

## 8. Reuse Risk
22. **Which parts can be reused safely?** The Keras model definition for the Multi-Patch Subnet.
23. **Which parts should not be reused?** The saliency detection code, due to the non-commercial license restriction.
24. **Would it help build our A-LAMP-paper-AVA-v2_graph_lite?** No, because it lacks the Layout-Aware Subnet and Graph logic.
25. **Is it worth downloading weights later?** No, as they are unofficial and incomplete.
26. **Is it worth porting any code?** No. Our local implementation is more robust and complete.

## 9. Comparison With Our v0/v1 Work
Our local implementation (`src/models/alamp_paper_ava.py`) already provides a cleaner, modern TensorFlow/Keras implementation of the Multi-Patch Subnet (using Attention-weighted pooling instead of simple Max/Avg, or fixed sequence features). The external repo completely lacks the Layout-Aware Subnet (0 bytes), whereas our `v0_b` variant at least provides a placeholder dense network for layout features and a global view branch. The external repo provides no value over our existing local code.

## 10. Decision
**D. Ignore external repo and build our own detector/graph pipeline.**

## 11. Next Step Recommendation
Since the external repository is incomplete, unofficial, and carries licensing risks for the saliency pipeline, we should proceed with building our own object detection and attribute graph pipeline using modern, unencumbered tools (e.g., standard object detectors) to fulfill the Layout-Aware Subnet requirements.