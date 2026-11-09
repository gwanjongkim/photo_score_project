# A-cut: Mobile Photo Evaluation App (Backend & Research)

This repository contains the backend research, training, evaluation, and model export code for the **A-cut capstone project**. A-cut is designed to evaluate mobile photos across multiple dimensions, including technical quality, overall aesthetics, and color composition.

**Note:** This repository focuses on the model engineering, distillation, and export pipelines. It is a research and capstone deliverable repository, not the primary production Flutter application code. Some results are experimental candidates supporting the capstone evidence.

## 1. Project Overview
The A-cut backend pipeline is responsible for:
1.  **Dataset Curation & Preprocessing:** Handling datasets like AVA, AADB, FLIVE, and KonIQ.
2.  **Teacher Model Inference:** Running heavy vision-language models (e.g., VILA, Gemma, Gemini API) and state-of-the-art IQA models (TOPIQ, MUSIQ) to generate high-quality pseudo-labels.
3.  **Student Distillation:** Training lightweight, mobile-friendly models (e.g., MobileNetV3-based) using the teacher's labels.
4.  **TFLite Export:** Converting trained PyTorch/TensorFlow models to `.tflite` format for integration into the Flutter mobile app.
5.  **Parity Benchmarking:** Ensuring models behave identically on the WSL backend and Android device.

## 2. A-cut Pipeline
Our pipeline separates technical quality (blur, noise, exposure) from subjective aesthetics.

### Technical Quality Models (IQA)
*   **Teacher Models:** TOPIQ, MUSIQ, KonIQ.
*   **Distillation Target:** MobileNetV3, explicitly trained to regress technical quality scores.
*   **Key Scripts:** Located in `scripts/technical_iqa/`.

### Aesthetic Quality Models
*   **Teacher Models:** NIMA, A-LAMP, RGNet, and custom multimodal setups (Gemma 4 E4B, VILA 1.5).
*   **Ensemble Approach:** Combining global context, local patches, and composition graphs.
*   **Key Scripts:** Located in `src/train/` and `src/eval/`.

### Color Aesthetic Models
*   **Base Model:** ICAA (Image Color Aesthetics Assessment).
*   **Engineering Effort:** Reimplemented PyTorch ICAA in native TensorFlow (`experiments/icaa_tf_native/`) to achieve a stable, mobile-ready TFLite export, bypassing ONNX opset limitations.

## 3. Mobile / TFLite Export
Deploying to mobile required significant effort in model quantization and conversion:
*   Exports are tracked in `exports/tflite/`.
*   Parity tests (`outputs/c6_android_wsl_parity*`) guarantee that the math remains identical after floating-point/int8 conversion and Android NNAPI/CPU delegation.

## 4. Benchmark Protocol & Key Results
*   Extensive benchmarking was performed against established datasets (FLIVE, AVA).
*   Parity scripts (`run_wsl_c6_parity_with_orientation.py`) verified inference stability.
*   For detailed benchmark artifacts and logs, refer to the mapped outputs in `docs/capstone_artifact_map_20260605.md`.

## 5. Reproducibility & Repository Structure
```
photo_score_project/
├── configs/            # Experiment YAML/JSON configs
├── docs/               # Architecture docs, audits, and capstone reports
├── experiments/        # Isolated sub-projects (e.g., TF Native ICAA)
├── exports/            # Allowed TFLite deliverables for mobile
├── scripts/            # Benchmarking and parity executable scripts
├── src/                # Core Python model architectures and training loops
└── tools/              # Utilities for patch extraction and threshold tuning
```

## 6. What is NOT Included in GitHub
Due to GitHub file size limits and dataset licensing constraints, the following are ignored via `.gitignore`:
*   **Datasets:** Raw and processed images (`data/`).
*   **Large Checkpoints:** Original weights for Gemma, VILA, NIMA, etc. (`checkpoints/`, `models/`).
*   **Virtual Environments:** `.venv*`
*   **Heavy Outputs:** Huge `.safetensors`, `.h5`, or massive CSV benchmark files (>50MB).

## 7. How to Run Selected Scripts
Ensure your Python environment (Python 3.10+) is set up according to `requirements.txt`.
```bash
# Example: Run a technical IQA benchmark
python -m scripts.technical_iqa.benchmark_technical_iqa_signals_20260604

# Example: Run parity check
python run_wsl_c6_parity_with_orientation.py
```