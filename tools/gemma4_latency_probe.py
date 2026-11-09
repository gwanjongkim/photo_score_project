from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gemma_distill.common import PROJECT_ROOT, repo_relative_path, resolve_repo_path
from src.gemma_distill.evaluate_gemma import extract_json_payload, validate_teacher_json
from src.gemma_distill.predict_gemma import normalize_messages


DEFAULT_MANIFEST = "outputs/gemini_api_teacher_aesthetic3_20260427_revised/test.jsonl"
DEFAULT_BASE_MODEL = "google/gemma-4-E4B-it"
DEFAULT_HF_HOME = (
    "outputs/gemini_api_teacher_aesthetic3_20260427_revised/"
    "gemma4_train_run_282/hf_cache"
)


class FirstTokenTimer:
    """Minimal Hugging Face streamer that records the first generated token callback."""

    def __init__(self) -> None:
        self._saw_prompt = False
        self.first_token_perf_counter: float | None = None
        self.token_callback_count = 0

    def put(self, value: Any) -> None:
        if not self._saw_prompt:
            self._saw_prompt = True
            return
        if self.first_token_perf_counter is None:
            self.first_token_perf_counter = time.perf_counter()
        self.token_callback_count += 1

    def end(self) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-only Gemma 4 E4B generation latency probe.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter_dir")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--row_index", type=int, default=0)
    parser.add_argument("--image")
    parser.add_argument("--selection_state", default="selected")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--hf_home", default=DEFAULT_HF_HOME)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--device_map", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--label", default="gemma4_e4b")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def set_hf_env(hf_home: str | None, *, local_files_only: bool) -> dict[str, str | bool | None]:
    if hf_home:
        hf_home_path = resolve_repo_path(hf_home) or Path(hf_home).expanduser().resolve(strict=False)
        os.environ["HF_HOME"] = str(hf_home_path)
        os.environ["HF_HUB_CACHE"] = str(hf_home_path / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home_path / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home_path / "transformers")
        os.environ["HF_MODULES_CACHE"] = str(hf_home_path / "modules")
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return {
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "local_files_only": local_files_only,
    }


def load_jsonl_row(path: Path, row_index: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == row_index:
                return json.loads(line)
    return None


def fallback_messages(selection_state: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the Gemma 4 E4B A-cut student. Return a JSON object with the fields: "
                "comment_type, short_reason, detailed_reason, and comparison_reason. Use visible evidence only. "
                "Do not mention internal guidance, scores, numbers, thresholds, rankings, percent bands, model names, "
                "bullets, or extra text."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Use the image first. The internal guidance controls only the comment style.\n"
                        f"Internal selection state: {selection_state}.\n"
                        "Write visible evidence for why this frame made or missed the A-cut.\n"
                        "The comment_type must be appropriate for the selection_state.\n"
                        "Return a single JSON object."
                    ),
                },
            ],
        },
    ]


def prompt_from_row_or_fallback(processor: Any, row: dict[str, Any] | None, selection_state: str) -> tuple[str, str]:
    if row and row.get("messages"):
        messages = normalize_messages(list(row["messages"])[:2])
        source = "manifest_messages_first_two"
    else:
        messages = fallback_messages(selection_state)
        source = "fallback_probe_prompt"
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, source


def resolve_image_path(image_arg: str | None, row: dict[str, Any] | None) -> Path | None:
    if image_arg:
        return resolve_repo_path(image_arg) or Path(image_arg).expanduser().resolve(strict=False)
    if row:
        image_value = row.get("image") or row.get("image_path")
        if image_value:
            return resolve_repo_path(str(image_value)) or Path(str(image_value)).expanduser().resolve(strict=False)
    return None


def nvidia_smi_snapshot() -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception as exc:
        return {"available": False, "error": f"{exc.__class__.__name__}: {exc}"}
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 4:
        return {"available": False, "raw": completed.stdout.strip()}
    return {
        "available": True,
        "name": parts[0],
        "memory_total_mib": int(parts[1]),
        "memory_used_mib": int(parts[2]),
        "memory_free_mib": int(parts[3]),
    }


def torch_gpu_snapshot(torch_module: Any, label: str) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {"label": label, "available": False}
    torch_module.cuda.synchronize()
    return {
        "label": label,
        "available": True,
        "torch_allocated_bytes": int(torch_module.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch_module.cuda.memory_reserved()),
        "torch_max_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "nvidia_smi": nvidia_smi_snapshot(),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_path = resolve_repo_path(args.output_json)
    manifest_path = resolve_repo_path(args.manifest)
    adapter_dir = resolve_repo_path(args.adapter_dir) if args.adapter_dir else None
    if output_path is None:
        raise ValueError("Could not resolve output_json.")
    if args.adapter_dir and (adapter_dir is None or not adapter_dir.exists()):
        raise FileNotFoundError(f"Adapter dir not found: {args.adapter_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hf_environment = set_hf_env(args.hf_home, local_files_only=bool(args.local_files_only))
    row = load_jsonl_row(manifest_path, args.row_index) if manifest_path else None
    image_path = resolve_image_path(args.image, row)
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Probe image not found: {image_path}")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if adapter_dir is not None:
        from peft import PeftModel
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("CUDA was required for this latency probe, but torch.cuda.is_available() is false.")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    gpu_snapshots: list[dict[str, Any]] = [torch_gpu_snapshot(torch, "start")]

    load_start = time.perf_counter()
    processor_source = str(adapter_dir) if adapter_dir is not None else args.base_model
    processor = AutoProcessor.from_pretrained(processor_source, local_files_only=bool(args.local_files_only))

    model_kwargs: dict[str, Any] = {"local_files_only": bool(args.local_files_only)}
    quantization: dict[str, Any] = {"enabled": False}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if args.device_map == "cuda":
            if device != "cuda":
                raise RuntimeError("--device_map cuda was requested, but CUDA is not available.")
            model_kwargs["device_map"] = {"": 0}
        elif args.device_map == "cpu":
            model_kwargs["device_map"] = {"": "cpu"}
        else:
            model_kwargs["device_map"] = "auto"
        quantization = {
            "enabled": True,
            "type": "bitsandbytes_nf4_4bit",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
            "device_map": args.device_map,
        }
    else:
        model_kwargs["dtype"] = dtype

    model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)
    if not args.load_in_4bit:
        model = model.to(device)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    hf_device_map = getattr(model, "hf_device_map", None)
    if args.require_cuda and isinstance(hf_device_map, dict):
        non_cuda_entries = {
            key: str(value)
            for key, value in hf_device_map.items()
            if str(value) not in {"0", "cuda", "cuda:0"}
        }
        if non_cuda_entries:
            raise RuntimeError(f"CUDA was required, but model device map contains non-CUDA entries: {non_cuda_entries}")
    load_seconds = time.perf_counter() - load_start
    gpu_snapshots.append(torch_gpu_snapshot(torch, "after_load"))

    prompt, prompt_source = prompt_from_row_or_fallback(
        processor,
        row,
        str(row.get("selection_state") if row else args.selection_state),
    )
    with Image.open(str(image_path)) as image:
        image_rgb = image.convert("RGB")
    inputs = processor(text=prompt, images=image_rgb, return_tensors="pt")
    input_summary = {
        key: list(value.shape) if hasattr(value, "shape") else str(type(value))
        for key, value in inputs.items()
    }
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    input_token_count = int(inputs["input_ids"].shape[1])

    streamer = FirstTokenTimer()
    generation_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max(1, int(args.max_new_tokens)),
            do_sample=False,
            streamer=streamer,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start
    gpu_snapshots.append(torch_gpu_snapshot(torch, "after_generate"))

    generated_ids = output_ids[0][input_token_count:]
    generated_token_count = int(generated_ids.shape[0])
    raw_prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()
    prediction_json, parse_error = extract_json_payload(raw_prediction)
    first_token_latency_seconds = (
        streamer.first_token_perf_counter - generation_start
        if streamer.first_token_perf_counter is not None
        else None
    )

    payload = {
        "label": args.label,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, *sys.argv],
        "cwd": str(PROJECT_ROOT),
        "base_model": args.base_model,
        "adapter_dir": repo_relative_path(adapter_dir) if adapter_dir is not None else None,
        "manifest": repo_relative_path(manifest_path) if manifest_path is not None else None,
        "row_index": args.row_index,
        "row_id": row.get("id") if row else None,
        "image_path": str(image_path),
        "prompt_source": prompt_source,
        "max_new_tokens": int(args.max_new_tokens),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "quantization": quantization,
        "model_hf_device_map": hf_device_map,
        "hf_environment": hf_environment,
        "timing": {
            "load_seconds": round(load_seconds, 6),
            "first_token_latency_seconds": (
                round(first_token_latency_seconds, 6) if first_token_latency_seconds is not None else None
            ),
            "generation_seconds": round(generation_seconds, 6),
            "generated_token_count": generated_token_count,
            "tokens_per_second": round(generated_token_count / generation_seconds, 6) if generation_seconds else None,
            "streamer_token_callback_count": streamer.token_callback_count,
        },
        "processor_input_summary": input_summary,
        "gpu_snapshots": gpu_snapshots,
        "raw_prediction": raw_prediction,
        "prediction_json": prediction_json,
        "parse_error": parse_error,
        "validation": validate_teacher_json(prediction_json, row.get("selection_state") if row else None),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "timing": payload["timing"]}, indent=2))


if __name__ == "__main__":
    main()
