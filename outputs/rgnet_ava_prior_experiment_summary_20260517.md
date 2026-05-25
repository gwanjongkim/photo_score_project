# RGNet AVA-prior Experiment Summary

## 1. Overview
This report summarizes the recent experiments aimed at improving the ranking performance of the paper-oriented RGNet implementation by leveraging large-scale pre-training on the AVA dataset. Despite successfully implementing the transfer learning pipeline, both AVA-prior fine-tuning variants underperformed compared to the existing ImageNet-initialized baseline.

## 2. Model Benchmarks

| Model Variant | Initialization | Aggregation | SRCC | PLCC | MAE | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Current Baseline** | **ImageNet** | **Mean** | **0.6819** | **0.6878** | **0.1198** | **KEEP** |
| AVA-prior Fixed | AVA Classification | LSE r4 | 0.6558 | 0.6700 | 0.1222 | Discard |
| AVA-prior Mean | AVA Classification | Mean | 0.6493 | 0.6624 | 0.1223 | Discard |

## 3. Key Findings
- **Technical Validation**: The weight-loading failure observed in the initial fine-tuning attempt was successfully resolved. The final experiments correctly loaded 643 variables across 5 major model components from the AVA classification weights.
- **Aggregation Strategy**: Mean aggregation continues to show superior ranking correlation compared to LSE aggregation for this specific graph-based architecture.
- **Pre-training Paradox**: Counter-intuitively, initializing the Mean-aggregation model from the binary AVA classification task resulted in a significant regression (-0.0326 SRCC) compared to starting from general ImageNet features. This suggests that the features learned for high/low aesthetic classification may not align perfectly with the continuous ranking task of AADB.

## 4. Final Decision
The current deployment candidate, **`models/aesthetic/rgnet_paper_aadb_fp16.tflite`**, remains the project record-holder and will not be replaced by the AVA-prior models. 

## 5. Formal Terminology
These experiments and implementations should be referred to as **“paper-oriented RGNet experiments/implementations”** rather than an official or exact reproduction of the original RGNet paper, as they utilize a custom Keras 3 port and project-specific training protocols.
