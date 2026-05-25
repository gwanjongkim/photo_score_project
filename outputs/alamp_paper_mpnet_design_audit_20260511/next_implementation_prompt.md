Implement Stage 1 and 2 of the A-LAMP-paper-MPNet validation plan: The Offline Patch Selector.

Do not modify existing practical A-LAMP models, Flutter code, or RGNet.
Do not train the model yet.

Create a new file: `src/datasets/alamp_paper_patch_selector.py`
It should act as an offline script to approximate the A-LAMP paper's adaptive patch selection and save the bounding boxes to JSONL files.

Requirements:
- Input: A CSV manifest with `image_path` and `mean_score`.
- Process: Read images at native resolution.
- Candidate Generation: Generate candidate 224x224 patches at multiple scales (e.g., 0.22, 0.35, 0.5 of the short edge).
- Saliency Approximation: Use OpenCV (cv2) or NumPy to compute a synthetic saliency score for each candidate. Since the exact paper saliency is not found locally, use a non-speculative approximation (e.g. combining Sobel edge density and local variance). Explicitly comment that this is a "saliency approximation".
- Diversity/Overlap Constraint: Iteratively select 5 patches per image. Pick the highest saliency patch first, then apply an IoU penalty to remaining candidates to enforce spatial diversity and overlap constraints.
- Output: Write to a JSONL file where each line contains:
  - `image_path`
  - `mean_score`
  - `boxes_norm_xyxy` (a list of 5 bounding boxes, normalized [0, 1])

Create a bash script `tools/run_alamp_paper_mpnet_patch_generation.sh` that:
1. Runs a smoke test of `src/datasets/alamp_paper_patch_selector.py` on 10 images from `outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv` and outputs to `outputs/alamp_paper_mpnet_design_audit_20260511/smoke_patches.jsonl`.
2. Upon success, processes the full 1024 subsets (train/val/test) from `outputs/alamp_object_graph_subset_20260511/subsets/` and saves them to `outputs/alamp_paper_mpnet_design_audit_20260511/subsets/`.

Run the bash script to verify execution and provide a summary of the smoke JSONL output.