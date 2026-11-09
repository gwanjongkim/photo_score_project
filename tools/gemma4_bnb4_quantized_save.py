from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gemma_distill.common import PROJECT_ROOT, repo_relative_path, resolve_repo_path


DEFAULT_HF_HOME = (
    "outputs/gemini_api_teacher_aesthetic3_20260427_revised/"
    "gemma4_train_run_282/hf_cache"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save a local-only Gemma 4 model as a BitsAndBytes 4-bit HF artifact.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hf_home", default=DEFAULT_HF_HOME)
    parser.add_argument("--device_map", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--require_cuda", action="store_true")
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


def dependency_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": f"{exc.__class__.__name__}: {exc}"}
    return {"available": True, "version": getattr(module, "__version__", None)}


def file_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for item in sorted(path.rglob("*")):
        if item.is_file():
            rows.append(
                {
                    "path": repo_relative_path(item),
                    "bytes": item.stat().st_size,
                }
            )
    return rows


def dir_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quantize_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    if output_dir is None:
        raise ValueError("Could not resolve output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_environment = set_hf_env(args.hf_home, local_files_only=bool(args.local_files_only))

    started = time.perf_counter()
    payload: dict[str, Any] = {
        "status": "started",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, *sys.argv],
        "cwd": str(PROJECT_ROOT),
        "source_model": args.model,
        "output_dir": repo_relative_path(output_dir),
        "format": "huggingface_transformers_bitsandbytes",
        "quantization_type": "nf4_4bit_double_quant",
        "runtime_required": "Python + PyTorch + Transformers + BitsAndBytes CUDA",
        "android_flutter_realistic": False,
        "android_blocker": "BitsAndBytes is a CUDA/PyTorch runtime path, not a LiteRT/TFLite, MediaPipe, or llama.cpp Android artifact.",
        "hf_environment": hf_environment,
        "dependencies": {
            name: dependency_status(name)
            for name in ("torch", "transformers", "bitsandbytes", "accelerate", "safetensors")
        },
    }
    write_summary(output_dir, payload)

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        cuda_available = bool(torch.cuda.is_available())
        payload["torch_cuda"] = {
            "available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
        if args.require_cuda and not cuda_available:
            raise RuntimeError("CUDA was required for 4-bit quantized save, but torch.cuda.is_available() is false.")

        if args.device_map == "cuda":
            if not cuda_available:
                raise RuntimeError("--device_map cuda was requested, but CUDA is not available.")
            device_map: str | dict[str, int | str] = {"": 0}
        elif args.device_map == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = "auto"

        load_start = time.perf_counter()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=bool(args.local_files_only))
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map=device_map,
            local_files_only=bool(args.local_files_only),
        )
        payload["load_seconds"] = round(time.perf_counter() - load_start, 6)
        payload["model_hf_device_map"] = getattr(model, "hf_device_map", None)

        save_start = time.perf_counter()
        model.save_pretrained(str(output_dir), safe_serialization=True)
        processor.save_pretrained(str(output_dir))
        payload["save_seconds"] = round(time.perf_counter() - save_start, 6)
        payload["status"] = "completed"
        payload["artifact_bytes"] = dir_size_bytes(output_dir)
        payload["files"] = file_inventory(output_dir)
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{exc.__class__.__name__}: {exc}"
    finally:
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        write_summary(output_dir, payload)

    print(json.dumps({key: payload.get(key) for key in ("status", "output_dir", "artifact_bytes", "error")}, indent=2))
    if payload["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
