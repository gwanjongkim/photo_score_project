# Salient Object Detector Audit For A-LAMP-paper-MPNet

## 1. Environment
- **Project Root**: `/home/omen_pc1/photo_score_project`
- **Python**: 3.12.3
- **Venv**: `.venv_gpu` confirmed with `torch 2.5.1+cu121` and `tensorflow` installed.
- **CUDA**: Active on **NVIDIA GeForce RTX 4070 SUPER**.
- **Disk Space**: ~626GB available.
- **Local Availability**: OpenCV and standard Python image libraries (PIL, numpy) are present. No specialized salient detection models currently exist in the `models/` directory.

## 2. Why V1/V2 Patch Selectors Failed
- **V1 (Sobel + Variance)**: Primarily detected high-frequency textures (grass, trees, noise) rather than semantically meaningful subjects. Manual inspection showed it missed the "core" saliency in 40/50 images.
- **V2 (Spectral Residual)**: Improved spatial diversity (61% IoU reduction) but actually **worsened** subject coverage (Object coverage dropped from 7.8% to 5.8%). The FFT-based approach often highlighted edges or background patterns instead of the actual salient subject in complex AVA photos.

## 3. Candidate Detector Matrix
| Candidate | License | Expected Quality | Feasibility | Integration |
| :--- | :--- | :--- | :--- | :--- |
| **U²-Net** | Apache 2.0 | High | High | Easy (PyTorch wrapper) |
| **BASNet** | Apache 2.0 | High | Medium | Medium |
| **IS-Net (DIS)**| Apache 2.0 | Very High | High | Easy (PyTorch wrapper) |
| **OpenCV** | BSD | Low | High | Rejected (already tested) |

## 4. License and Dependency Risk
- **U²-Net** is released under **Apache License 2.0**, which is safe for research and capstone use. Redistribution of pretrained weights is generally permitted if attribution is maintained.
- **IS-Net** (Dichotomous Image Segmentation) is also **Apache 2.0**.
- **Risks**: The primary dependency is `torch`, which is already available in the environment. No conflicting dependencies (like Caffe or old Python versions) were found.

## 5. Runtime Feasibility
- **GPU Speed**: On an **RTX 4070 SUPER**, U²-Net or IS-Net should process images at high FPS (likely >10-20 images per second).
- **Disk Usage**: Weights are typically <200MB. Output saliency maps for the 1024 subset will be negligible (e.g., compressed PNGs or small NPZ files).
- **Scale**: The approach is highly feasible for the 1024 subset and will scale easily to the full AVA dataset (~250k images) given the GPU power.

## 6. Integration Plan For Patch Selector V3
1. **Model Management**: Download U²-Net weights to `models/saliency/u2net.pth`.
2. **Offline Generation**: Run a script to generate grayscale masks for the 1024 train/val/test images.
3. **Scoring Logic**: Update `src/datasets/alamp_paper_patch_selector.py` to support `v3`. It will load the offline masks and calculate the average saliency within each candidate box.
4. **Diversity Logic**: Retain the V2 weighted objective (Saliency + Pattern Distance + Spatial Distance).
5. **Validation**: Use 50 visualizations and YOLO object coverage to prove subject capture.

## 7. Recommended Detector
**U²-Net**
- **Reasoning**: It is the most robust and widely used model for general salient object detection. Its architecture is well-understood, and multiple pretrained versions (standard and lightweight) allow for flexibility in case of memory constraints (though unlikely on an RTX 4070).

## 8. Fallback Detector
**IS-Net (DIS)**
- **Reasoning**: If U²-Net masks are too coarse, IS-Net provides state-of-the-art fine-grained masks for dichotomous segmentation.

## 9. What Not To Claim
- This is **NOT** a "paper-faithful reproduction" of A-LAMP's saliency model (which used a different, proprietary, or older saliency algorithm).
- It remains a **salient-object-guided Patch Selector V3 approximation**.

## 10. Next Patch Selector V3 Prompt
*See `next_patch_selector_v3_prompt.md` in the audit directory.*
