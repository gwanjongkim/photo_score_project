# External PaQ-2-PiQ Official Split Search

## 1. Summary
**A. Official split file found** (Standardized by community/toolbox)
While the original authors (Ying et al., 2020) described a "random split", subsequent benchmarking work (like MUSIQ) and toolboxes (like PyIQA) have standardized a specific split of **30,000 (Train) / 7,700 (Val) / 1,800 (Test)** images to ensure reproducibility. This split is available in the `meta_info_FLIVEDataset.csv` metadata file.

## 2. Sources Checked
- **https://github.com/baidut/PaQ-2-PiQ**: Official repository. Does not contain a static split file; uses `valid_pct=0.2` for random splitting at runtime.
- **https://github.com/chaofengc/IQA-PyTorch**: Near-official repository (PyIQA). Refers to `meta_info_FLIVEDataset.csv` and uses standardized splits.
- **https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets-metainfo**: Hosting site for the standardized metadata. **Confirmed presence of `meta_info_FLIVEDataset.csv`.**
- **UT LIVE Download Page**: Provides `All_Labels.csv` but no split indicators.

## 3. Candidate Split Files or Scripts
- **File:** `meta_info_FLIVEDataset.csv`
- **URL:** [Hugging Face Link](https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets-metainfo/blob/main/meta_info_FLIVEDataset.csv)
- **Source Type:** Near-official (IQA-Toolbox/PyIQA).
- **File Type:** CSV
- **Content:** Contains `official_split` column with `train`, `val`, and `test` labels.
- **Counts:** 30,000 / 7,700 / 1,800. Matches MUSIQ paper protocol.
- **Reproducibility:** YES. This is the primary file used by researchers to reproduce the paper's benchmark.

## 4. Official GitHub Notebook/Script Findings
- `database_prep.ipynb` in the official repo is used for initial dataset assembly (sampling patches, organizing files) but does not define the final benchmark split.
- The `fastiqa` library used by the authors defaults to a random 20% validation split if no other is provided.

## 5. UT LIVE / Dataset Page Findings
- The official UT LIVE page hosts the images and the `All_Labels.csv` file. It does not provide the 30k/7.7k/1.8k split metadata directly in the public download zip.

## 6. Decision
**B. Reconstruct split from official script/notebook or use Toolbox Metadata**
The most reliable way to reproduce the MUSIQ/PaQ-2-PiQ paper protocol is to use the `meta_info_FLIVEDataset.csv` file from the IQA-Toolbox, which has already mapped the "official" random split into a static CSV format.

## 7. Next Action
Ask Codex to download/build the paper-protocol manifest using the mappings from `meta_info_FLIVEDataset.csv` (Hugging Face) to align the local PaQ-2-PiQ dataset with the 30k/7.7k/1.8k protocol.

## 8. Forbidden Actions
- No custom splits were invented.
- Confirmed that the local 80/10/10 split is NOT the paper protocol.
- No training or modification of raw files occurred.
