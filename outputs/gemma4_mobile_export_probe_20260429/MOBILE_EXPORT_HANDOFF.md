# Gemma 4 E4B Mobile Export Handoff

Date: 2026-04-29

## Scope

Teacher dataset expansion is paused. No Gemini API calls, Batch API jobs, new teacher rows, new teacher experiments, or training runs were used for this handoff.

This package focuses on local model readiness, adapter merge, WSL CUDA latency probing, and practical mobile export status for Gemma 4 E4B.

## Source Artifacts

- Base model: `google/gemma-4-E4B-it`
- Local base cache: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/hf_cache/hub/models--google--gemma-4-E4B-it`
- Base cache size: 16024791416 bytes, about 15 GB
- Fine-tuned adapter: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/final_adapter`
- Adapter size: 67160340 bytes, about 65 MB
- Adapter config: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/final_adapter/adapter_config.json`
- Previous export summary: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/export_gemma4_282/export_summary.json`
- Previous export status: adapter only, no quantized export completed

## Exported In This Package

### Merged Fine-Tuned HF Model

- Type: fine-tuned Gemma 4 E4B, LoRA adapter merged into base
- Format: Hugging Face Transformers safetensors directory, BF16 config
- Path: `outputs/gemma4_mobile_export_probe_20260429/merged_gemini_teacher_282`
- Directory size: 15914673483 bytes, about 15 GB
- Main model file: `outputs/gemma4_mobile_export_probe_20260429/merged_gemini_teacher_282/model.safetensors`
- Main model file size: 15882477500 bytes
- Runtime required: Python, PyTorch, Transformers
- Android/Flutter realistic: no, not directly. This is a desktop/server HF artifact.

### Quantized HF Candidate

- Type: fine-tuned merged model loaded and saved with BitsAndBytes NF4 4-bit double quantization
- Format: Hugging Face Transformers + BitsAndBytes quantized safetensors directory
- Path: `outputs/gemma4_mobile_export_probe_20260429/quantized_bnb4_merged_gemini_teacher_282`
- Directory size: 9333880849 bytes, about 8.7 GB
- Main model file: `outputs/gemma4_mobile_export_probe_20260429/quantized_bnb4_merged_gemini_teacher_282/model.safetensors`
- Main model file size: 9301680673 bytes
- Runtime required: Python, PyTorch, Transformers, BitsAndBytes CUDA
- Android/Flutter realistic: no. BitsAndBytes is not a LiteRT, TFLite, MediaPipe, or llama.cpp Android artifact.
- Known extra runtime requirement on this WSL machine: set `LD_LIBRARY_PATH=/home/omen_pc1/photo_score_project/.venv_vila/lib/python3.12/site-packages/nvidia/cu13/lib`

## Latency Probe Results

### Base Model Probe

- Result file: `outputs/gemma4_mobile_export_probe_20260429/base_latency_probe.json`
- Model: `google/gemma-4-E4B-it`
- Quantization during probe: BitsAndBytes NF4 4-bit, CUDA
- Image input: yes, existing test manifest image `/home/omen_pc1/ava/images/289024.jpg`
- Prompt source: first two messages from `outputs/gemini_api_teacher_aesthetic3_20260427_revised/test.jsonl`
- `max_new_tokens`: 80
- Load time: 7.667396 s
- First token latency: 1.179458 s
- Generation time: 11.547373 s
- Generated tokens: 80
- Decode speed: 6.927983 tok/s
- GPU memory after load: 9642 MiB used on RTX 4070 SUPER
- GPU memory after generation: 10190 MiB used
- Output validity: JSON was incomplete at 80 tokens

### Merged Fine-Tuned Probe

- Result file: `outputs/gemma4_mobile_export_probe_20260429/merged_latency_probe.json`
- Model: `outputs/gemma4_mobile_export_probe_20260429/merged_gemini_teacher_282`
- Quantization during probe: BitsAndBytes NF4 4-bit, CUDA
- Image input: yes, same existing test manifest image
- `max_new_tokens`: 80
- Load time: 16.358292 s
- First token latency: 1.127054 s
- Generation time: 12.044716 s
- Generated tokens: 80
- Decode speed: 6.641917 tok/s
- GPU memory after load: 9642 MiB used on RTX 4070 SUPER
- GPU memory after generation: 10190 MiB used
- Output validity: JSON was incomplete at 80 tokens

## Prompt Format

Use the existing two-message format from the Gemma teacher manifest:

```text
System:
You are the Gemma 4 E4B A-cut student. Return a JSON object with the fields: comment_type, short_reason, detailed_reason, and comparison_reason. Use visible evidence only. Do not mention internal guidance, scores, numbers, thresholds, rankings, percent bands, model names, bullets, or extra text.

User:
[image]
Use the image first. The internal guidance controls only the comment style.
Internal selection state: selected | near_miss | not_selected.
Write visible evidence for why this frame made or missed the A-cut.
The comment_type must be appropriate for the selection_state.
Return a single JSON object.
```

Recommended initial generation settings:

- `max_new_tokens`: 128 for first Android latency test
- Increase to 160 only if JSON is truncated
- Use deterministic decoding first: temperature 0 or greedy
- Stop early on the closing JSON brace if the runtime supports stop conditions

## Android And Flutter Copy Guidance

Do not copy the HF or BitsAndBytes artifacts above into Flutter assets expecting them to run. They are not mobile runtime artifacts.

For a valid LiteRT-LM or MediaPipe-compatible artifact, use one of these Android handoff layouts:

```bash
mkdir -p android/app/src/main/assets/models
cp /path/to/model.litertlm android/app/src/main/assets/models/gemma4_e4b.litertlm
```

For direct device testing without packaging into Flutter assets:

```bash
adb shell rm -rf /data/local/tmp/llm
adb shell mkdir -p /data/local/tmp/llm
adb push /path/to/model.litertlm /data/local/tmp/llm/
```

Current package has no `.litertlm`, `.task`, `.tflite`, or `.gguf` file to copy.

## Runtime Recommendation

Recommended path for Galaxy S23 Ultra-class latency testing:

1. First test the official base Gemma 4 E4B LiteRT-LM artifact on device.
2. Use MediaPipe LLM Inference or LiteRT-LM, not the HF BitsAndBytes artifact.
3. If base latency is acceptable, then work on converting the merged fine-tuned HF model or an adapter path into a LiteRT-LM compatible artifact.

Why:

- Google AI Edge documents LiteRT as the on-device runtime for `.tflite` and `.litertlm` artifacts: https://ai.google.dev/edge/litert/overview
- MediaPipe LLM Inference Android is documented for high-end Android devices including Samsung S23 or later and uses `com.google.mediapipe:tasks-genai`: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android
- MediaPipe documents use of pre-converted LiteRT Community models or conversion through AI Edge Torch Generative Converter: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference
- The official LiteRT Community Gemma 4 E4B model card provides a `.litertlm` deployment artifact and mobile benchmark context: https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm

## Expected Latency Risk

The WSL 4-bit CUDA path is slower than expected for mobile-optimized LiteRT-LM: about 6.6 to 6.9 tok/s on RTX 4070 SUPER through Transformers and BitsAndBytes. This is not a direct Android predictor because the runtime stack differs.

The official LiteRT-LM Gemma 4 E4B model card reports a 3.65 GB mobile artifact and S26 Ultra GPU decode around 22.1 tok/s with 0.8 s TTFT. Galaxy S23 Ultra should be tested directly because it is older and may be lower. The current fine-tuned artifact is not in that optimized format.

## Known Blockers

- No mobile-ready fine-tuned artifact was produced.
- No LiteRT-LM, MediaPipe `.task`, TFLite, or GGUF export script exists in the current repo for Gemma 4 E4B.
- Installed environment lacks `mediapipe`, `ai_edge_torch`, `optimum`, `llama_cpp`, and llama.cpp command-line converters.
- Current quantized candidate requires CUDA BitsAndBytes and is about 8.7 GB, so it is not realistic for Flutter/Android integration.
- The 80-token probes truncated JSON. Android testing should use 128 to 160 max new tokens or runtime stop-on-JSON-close.
- The WSL BitsAndBytes CUDA path requires `LD_LIBRARY_PATH` to include `.venv_vila/lib/python3.12/site-packages/nvidia/cu13/lib`.
- Fine-tuned mobile deployment depends on a future or external conversion path from merged HF Gemma 4 E4B to LiteRT-LM/MediaPipe-compatible format.

## Commands Run

Base latency probe:

```bash
LD_LIBRARY_PATH=/home/omen_pc1/photo_score_project/.venv_vila/lib/python3.12/site-packages/nvidia/cu13/lib .venv_vila/bin/python tools/gemma4_latency_probe.py --label base_gemma4_e4b_bnb4_cuda --base_model google/gemma-4-E4B-it --hf_home outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/hf_cache --manifest outputs/gemini_api_teacher_aesthetic3_20260427_revised/test.jsonl --output_json outputs/gemma4_mobile_export_probe_20260429/base_latency_probe.json --max_new_tokens 80 --load_in_4bit --device_map cuda --require_cuda
```

Adapter merge:

```bash
.venv_vila/bin/python -m src.gemma_distill.merge_adapter --base_model google/gemma-4-E4B-it --adapter_dir outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/final_adapter --output_dir outputs/gemma4_mobile_export_probe_20260429/merged_gemini_teacher_282 --hf_home outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/hf_cache --local_files_only
```

Merged latency probe:

```bash
LD_LIBRARY_PATH=/home/omen_pc1/photo_score_project/.venv_vila/lib/python3.12/site-packages/nvidia/cu13/lib .venv_vila/bin/python tools/gemma4_latency_probe.py --label merged_gemma4_e4b_teacher282_bnb4_cuda --base_model outputs/gemma4_mobile_export_probe_20260429/merged_gemini_teacher_282 --hf_home outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/hf_cache --manifest outputs/gemini_api_teacher_aesthetic3_20260427_revised/test.jsonl --output_json outputs/gemma4_mobile_export_probe_20260429/merged_latency_probe.json --max_new_tokens 80 --load_in_4bit --device_map cuda --require_cuda
```

Quantized HF save:

```bash
LD_LIBRARY_PATH=/home/omen_pc1/photo_score_project/.venv_vila/lib/python3.12/site-packages/nvidia/cu13/lib .venv_vila/bin/python tools/gemma4_bnb4_quantized_save.py --model outputs/gemma4_mobile_export_probe_20260429/merged_gemini_teacher_282 --output_dir outputs/gemma4_mobile_export_probe_20260429/quantized_bnb4_merged_gemini_teacher_282 --hf_home outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/hf_cache --device_map cuda --require_cuda
```

## Exact Next Step

Use this next prompt for the Flutter/Android integration handoff:

```text
Build a minimal Android-only Gemma 4 E4B latency harness for Galaxy S23 Ultra-class testing using MediaPipe LLM Inference or LiteRT-LM. Do not use the HF BitsAndBytes artifacts. Start with the official LiteRT Community Gemma 4 E4B `.litertlm` base artifact, copy it to `/data/local/tmp/llm/`, run the A-cut JSON prompt with one image input if the runtime supports multimodal input, set max_new_tokens to 128, and report load time, first token latency, total generation time, tokens/sec, memory, and whether JSON closes. Keep this separate from `~/pozy_app` unless explicitly asked to integrate it there.
```
