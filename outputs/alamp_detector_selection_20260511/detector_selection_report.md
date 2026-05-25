# A-LAMP Detector Selection Report

## 1. Environment
- **OS:** WSL (Linux)
- **Python:** 3.12.3
- **Virtual Env:** `.venv_gpu`
- **GPU:** NVIDIA RTX 4070 SUPER (Currently detected by TensorFlow 2.20.0)
- **PyTorch:** Not currently installed in the environment.
- **Existing Detectors:** None found in the project.

## 2. Goal
Select a simple, robust, pretrained object detector to extract bounding boxes and classes for AVA images. These detections will be used to construct the Object/Global Attribute Graph for the A-LAMP-paper-AVA-v2 architecture.

## 3. Candidate Detector Matrix
- **Ultralytics YOLOv8/11:** Simplest API, minimal boilerplate, extremely fast on RTX 4070 SUPER, handles JSON output easily.
- **torchvision Faster R-CNN:** Requires writing custom loops, NMS, resizing logic, and output formatting.
- **Detectron2:** High risk of compilation issues in WSL/Windows environments; overkill for standard COCO detection.
- **RT-DETR:** Good, but YOLOv11 provides similar performance and is bundled in the same Ultralytics library.
- **GroundingDINO:** Slow, complex dependencies, unnecessary for generic object classes.

## 4. Dependency and License Risk
Ultralytics is AGPL-3.0 licensed. For internal dataset/artifact generation where the model code is not distributed as part of a commercial application, this is entirely safe. It has minimal dependency clashes.

## 5. GPU/Runtime Feasibility
The RTX 4070 SUPER will run YOLOv11/v8 at very high speeds, making processing the full 250k AVA dataset viable in a short timeframe once the pipeline is validated.

## 6. Output Format Suitability
Ultralytics provides a `Results` object that contains all required attributes:
- `boxes.cls` (class_id)
- `names` dictionary (class_name)
- `boxes.conf` (confidence)
- `boxes.xyxy` (bounding box)
- `orig_shape` (image width/height)

## 7. Recommended Detector
**Ultralytics YOLO** (`ultralytics` package). It is the simplest and most robust option.

## 8. 1024-image Subset Test Plan
1. Install PyTorch with CUDA support.
2. Install `ultralytics`.
3. Create an extraction script to load the YOLOv11m/YOLOv8m model.
4. Process a small 1024-image chunk of AVA and export to JSONL.

## 9. Next Codex Prompt
"Implement the bounding box extraction script using Ultralytics YOLOv11. Install dependencies, and run it on a 1024-image AVA subset to verify output format. Delay full AVA extraction until the graph logic is validated."
