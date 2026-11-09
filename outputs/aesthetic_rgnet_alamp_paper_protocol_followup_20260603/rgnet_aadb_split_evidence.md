# RGNet/AADB Split Evidence Audit

## Evidence Files
- **Official Metadata**: `data/raw/aadb/AADBinfo.mat`
- **Processing Logic**: `src/preprocess/make_aadb_csv.py`
- **Benchmark Split**: `data/processed/aadb/test.csv`
- **Predictions**: `outputs/aesthetic_mobile_paper_benchmark_20260603/full/rgnet_aadb/rgnet_predictions.csv`

## Audit Process & Findings
1.  **Direct Source Check**: `make_aadb_csv.py` explicitly loads the `AADBinfo.mat` file, which is the standard metadata source for the AADB dataset.
2.  **Split Identity**: The script uses the `testNameList` and `testScore` fields from the MAT file.
3.  **Parity Verification**: An intersection check between the filenames in `AADBinfo.mat` (`testNameList`) and `data/processed/aadb/test.csv` confirmed an **exact 1000/1000 image match**.
4.  **Row Count Consistency**: 
    - Original MAT test count: 1000
    - Processed CSV test count: 1000
    - Evaluated images: 1000
5.  **Metric Stability**: Recomputed metrics on the existing prediction CSV match the reported values exactly.

## Recomputed Metrics (Standard AADB Test Set)
- **Count**: 1000 images
- **SRCC**: 0.6647
- **PLCC**: 0.6708
- **MSE**: 0.0232
- **MAE**: 0.1219
- **Score Scale**: Ground Truth and Predictions both on 0..1 scale.

## Final Verdict
**RGNet/AADB can be claimed as a full standard AADB paper-comparable benchmark.** The evidence proves that the evaluation was performed on the official test split with no missing or duplicated samples, following standard preprocessing.
