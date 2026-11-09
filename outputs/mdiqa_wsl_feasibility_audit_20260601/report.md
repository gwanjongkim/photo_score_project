# MDIQA WSL Feasibility Audit Report

## 1. Executive Summary
MDIQA (Multi-dimensional Image Quality Assessment) is a strong candidate for a technical IQA teacher, with paper-reported SRCC numbers significantly exceeding the current production baseline on SPAQ (+0.055) and FLIVE (+0.073). Pretrained checkpoints are available on Google Drive. The environment setup in WSL is feasible via an isolated `.venv_mdiqa`, though it requires a specific patch to `open_clip` and lacks a local `nvcc` which may complicate any CUDA-accelerated restoration components (though likely not the core IQA inference). The primary risk is the **S-Lab Non-commercial License**, which may restrict commercial application of the distilled student.

## 2. Repository Status
*   **Source:** [YaoShunyu19/MDIQA](https://github.com/YaoShunyu19/MDIQA)
*   **Architecture:** Multi-dimensional heads (Technical, Aesthetic) with CLIP-based feature injection.
*   **Code Quality:** Research-grade, based on `pyiqa` and `BasicSR`.
*   **License:** **S-Lab License 1.0 (Non-commercial / Academic only)**.

## 3. Checkpoint Availability
*   **Status:** **Available**
*   **Link:** [Google Drive Folder](https://drive.google.com/drive/folders/1-RajAoPc-0v-WWQ9BnYTi-HxbpJNLZcl?usp=sharing)
*   **Models needed:**
    *   `MDIQA_SPAQ.pth` (or similar) for SPAQ evaluation.
    *   `MDIQA_KonIQ.pth` for KonIQ-10k.
    *   `MDIQA_FLIVE.pth` for FLIVE.

## 4. Environment Requirements
*   **Python:** 3.10 recommended (3.12.3 available in WSL).
*   **PyTorch:** `torch>=1.13`, `torchvision>=0.13`.
*   **Critical Dependency:** `open_clip` requires a manual `sed` patch on line 93 of `modified_resnet.py`.
*   **Isolated Env:** `.venv_mdiqa` (approx. 4-6GB including torch and dependencies).

## 5. WSL Compatibility Risks
*   **Missing `nvcc`**: The current WSL environment lacks `nvcc`. This will prevent compilation of custom CUDA ops in `BasicSR/basicsr/ops`. However, `inference_iqa.py` appears to rely on `pyiqa` and standard torch operations, which should run (albeit slower if custom kernels are missing, or potentially failing if they are strictly required for the backbone).
*   **Python 3.12**: Some older dependencies in `requirements.txt` (like `bitsandbytes` or specific `transformers` versions) might require minor version adjustments for Python 3.12 compatibility.

## 6. Minimum Smoke Test Plan
1.  Clone MDIQA repo.
2.  Create `.venv_mdiqa` and install `requirements.txt`.
3.  Clone and patch `open_clip` as per README instructions.
4.  Download one checkpoint (e.g., `MDIQA_SPAQ.pth`).
5.  Run `python inference_iqa.py` on a set of 5 local test images to verify score output.

## 7. Benchmark Evaluation Plan
1.  Map the existing benchmark manifest `outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv` to MDIQA inference inputs.
2.  Evaluate on **FLIVE**, **KonIQ-10k**, and **SPAQ**.
3.  Calculate SRCC/PLCC using the `quality_score_0_100` column as ground truth.
4.  Compare directly against:
    *   FLIVE: 0.574
    *   KonIQ-10k: 0.934
    *   SPAQ: 0.873

## 8. Go/No-Go Decision
**Decision: MDIQA_READY_FOR_SMOKE_SETUP**

*   **Go**: Significant performance headroom over current baseline. Checkpoints are accessible. Environment isolation is possible.
*   **Caution**: The Non-commercial license must be reviewed if the distillation target is for a commercial product. The missing `nvcc` is a technical risk that will be cleared during the smoke test.

## 9. Exact Next Commands

```bash
# 1. Setup isolated environment
python3 -m venv .venv_mdiqa
source .venv_mdiqa/bin/activate
pip install --upgrade pip

# 2. Clone and install MDIQA
git clone https://github.com/YaoShunyu19/MDIQA.git
cd MDIQA
pip install -r requirements.txt

# 3. Install patched open_clip
cd ..
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
sed -i '93s/return x\[0\]/return x/' src/open_clip/modified_resnet.py 
pip install -e .
cd ../MDIQA

# 4. Install MDIQA in develop mode
python3 setup.py develop
```

*Note: For the smoke test, manually download weights to `experiments/pretrained_models/` before running inference.*
