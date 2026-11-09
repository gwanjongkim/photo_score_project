from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.gemma_distill.common import PROJECT_ROOT, read_json, repo_relative_path, resolve_repo_path, write_json


DEFAULT_CONFIG = {
    "experiment_name": "gemma4_e4b_aesthetic3_smoke",
    "dataset_dir": "outputs/gemma4_e4b_aesthetic3_smoke",
    "train_manifest": "train.jsonl",
    "val_manifest": "val.jsonl",
    "test_manifests": {
        "test": "test.jsonl",
        "test_vila": "test_vila.jsonl",
    },
    "output_dir": "outputs/gemma4_e4b_aesthetic3_smoke/train_run",
    "model_name_or_path": "google/gemma-4-E4B-it",
    "processor_name_or_path": None,
    "base_adapter_dir": None,
    "allow_remote_model_download": False,
    "lora": {
        "enabled": True,
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": "all-linear",
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    },
    "training": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "logging_steps": 1,
        "logging_first_step": True,
        "save_steps": 10,
        "eval_steps": 10,
        "save_total_limit": 2,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "optim": "adamw_torch_fused",
        "seed": 42,
        "max_length": None,
        "resume_from_checkpoint": None,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "test_manifests":
            merged[key] = value
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight or run Gemma 4 E4B multimodal fine-tuning.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight_only", action="store_true")
    return parser


def dependency_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": f"{exc.__class__.__name__}: {exc}"}
    return {"available": True, "version": getattr(module, "__version__", None)}


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise ValueError("Could not resolve training config path.")
    override = read_json(resolved)
    return deep_merge(DEFAULT_CONFIG, override), resolved


def summarize_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0}
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for rows, _ in enumerate(handle, start=1):
            pass
    return {"exists": True, "rows": max(0, rows)}


def validate_manifest_targets(path: Path) -> dict[str, Any]:
    required = {"comment_type", "short_reason", "detailed_reason", "comparison_reason"}
    allowed_comment_types = {"selected_explanation", "near_miss_feedback", "rejection_reason"}
    summary: dict[str, Any] = {
        "exists": path.exists(),
        "rows": 0,
        "assistant_json_valid_rows": 0,
        "teacher_comment_json_valid_rows": 0,
        "missing_required_field_rows": 0,
        "invalid_comment_type_rows": 0,
    }
    if not path.exists():
        return summary
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            summary["rows"] += 1
            row = json.loads(line)
            target_payload: dict[str, Any] | None = None
            teacher_comment_json = row.get("teacher_comment_json")
            if teacher_comment_json:
                try:
                    parsed = json.loads(str(teacher_comment_json))
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    summary["teacher_comment_json_valid_rows"] += 1
                    target_payload = parsed
            messages = row.get("messages") or []
            if messages:
                try:
                    assistant_text = messages[-1]["content"][0]["text"]
                    assistant_payload = json.loads(assistant_text)
                except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                    assistant_payload = None
                if isinstance(assistant_payload, dict):
                    summary["assistant_json_valid_rows"] += 1
                    target_payload = assistant_payload
            if not isinstance(target_payload, dict) or required - set(target_payload):
                summary["missing_required_field_rows"] += 1
            elif target_payload.get("comment_type") not in allowed_comment_types:
                summary["invalid_comment_type_rows"] += 1
    summary["ready"] = (
        summary["exists"]
        and summary["rows"] > 0
        and summary["assistant_json_valid_rows"] == summary["rows"]
        and summary["missing_required_field_rows"] == 0
        and summary["invalid_comment_type_rows"] == 0
    )
    return summary


def ensure_hf_cache_env(output_dir: Path) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        for token_path in (
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        ):
            if token_path.exists():
                candidate = token_path.read_text(encoding="utf-8").strip()
                if candidate:
                    token = candidate
                    break

    hf_home = Path(os.environ.get("HF_HOME") or (output_dir / "hf_cache")).resolve(strict=False)
    hub_cache = hf_home / "hub"
    transformers_cache = hf_home / "transformers"
    modules_cache = hf_home / "modules"
    for path in (hf_home, hub_cache, transformers_cache, modules_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["HF_MODULES_CACHE"] = str(modules_cache)

    token_present = bool(token)
    if token and not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HF_TOKEN"] = token

    return {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hub_cache),
        "TRANSFORMERS_CACHE": str(transformers_cache),
        "HF_MODULES_CACHE": str(modules_cache),
        "huggingface_token_present": token_present,
    }


def gather_preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    dataset_dir = resolve_repo_path(config["dataset_dir"])
    output_dir = resolve_repo_path(config["output_dir"])
    if dataset_dir is None or output_dir is None:
        raise ValueError("Could not resolve dataset_dir or output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_env = ensure_hf_cache_env(output_dir)

    train_manifest = (dataset_dir / str(config["train_manifest"])).resolve(strict=False)
    val_manifest = (dataset_dir / str(config["val_manifest"])).resolve(strict=False)
    test_manifests = {
        split_name: (dataset_dir / str(path_value)).resolve(strict=False)
        for split_name, path_value in dict(config.get("test_manifests") or {}).items()
    }

    dependency_names = ["torch", "transformers", "datasets", "trl", "peft", "accelerate", "bitsandbytes"]
    dependencies = {name: dependency_status(name) for name in dependency_names}

    torch_cuda = {"available": False, "device_count": 0, "devices": []}
    try:
        import torch

        torch_cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception:
        pass

    model_name_or_path = str(config["model_name_or_path"])
    local_model_path = resolve_repo_path(model_name_or_path)
    is_local_model = local_model_path is not None and local_model_path.exists()
    blocked_reasons: list[str] = []

    manifest_summary = {
        "train": summarize_manifest(train_manifest),
        "val": summarize_manifest(val_manifest),
        "tests": {split_name: summarize_manifest(path) for split_name, path in test_manifests.items()},
    }
    target_validation = {
        "train": validate_manifest_targets(train_manifest),
        "val": validate_manifest_targets(val_manifest),
        "tests": {split_name: validate_manifest_targets(path) for split_name, path in test_manifests.items()},
    }
    if not manifest_summary["train"]["exists"]:
        blocked_reasons.append(f"Missing train manifest: {train_manifest}")
    if not manifest_summary["val"]["exists"]:
        blocked_reasons.append(f"Missing val manifest: {val_manifest}")
    if not target_validation["train"].get("ready"):
        blocked_reasons.append(f"Train manifest JSON teacher targets failed validation: {train_manifest}")
    if not target_validation["val"].get("ready"):
        blocked_reasons.append(f"Val manifest JSON teacher targets failed validation: {val_manifest}")

    for package_name in ("torch", "transformers", "datasets", "trl", "peft", "accelerate"):
        if not dependencies[package_name]["available"]:
            blocked_reasons.append(f"Missing dependency: {package_name}")

    transformers_supports_gemma4 = False
    model_access_check: dict[str, Any] = {"attempted": False}
    if dependencies["transformers"]["available"]:
        try:
            from transformers import AutoModelForImageTextToText  # noqa: F401
            from transformers import AutoProcessor
            from transformers.models.gemma4.configuration_gemma4 import Gemma4Config  # noqa: F401

            transformers_supports_gemma4 = True
            model_access_check["attempted"] = True
            AutoProcessor.from_pretrained(
                model_name_or_path,
                local_files_only=not bool(config.get("allow_remote_model_download", False)),
            )
            model_access_check["processor_load_ok"] = True
        except Exception as exc:
            if not transformers_supports_gemma4:
                blocked_reasons.append(f"Installed transformers build does not expose Gemma 4 image-text classes: {exc}")
            else:
                model_access_check["processor_load_ok"] = False
                model_access_check["error"] = f"{exc.__class__.__name__}: {exc}"
                blocked_reasons.append(f"Could not load Gemma 4 processor/model access preflight: {exc}")

    if not is_local_model and not bool(config.get("allow_remote_model_download", False)):
        blocked_reasons.append(
            "model_name_or_path points to a remote model id, but allow_remote_model_download=false and no local checkpoint exists."
        )
    if bool(config["quantization"].get("load_in_4bit", False)) and not dependencies["bitsandbytes"]["available"]:
        blocked_reasons.append("4-bit quantization requested but bitsandbytes is not available.")
    if bool(config["quantization"].get("load_in_4bit", False)) and not torch_cuda["available"]:
        blocked_reasons.append("4-bit QLoRA path requested but torch CUDA is not available in this WSL environment.")

    return {
        "experiment_name": config["experiment_name"],
        "config_path": repo_relative_path(config_path),
        "dataset_dir": repo_relative_path(dataset_dir),
        "output_dir": repo_relative_path(output_dir),
        "python_executable": sys.executable,
        "cwd": str(PROJECT_ROOT),
        "environment": {
            "HF_HOME": hf_env["HF_HOME"],
            "HF_HUB_CACHE": hf_env["HF_HUB_CACHE"],
            "TRANSFORMERS_CACHE": hf_env["TRANSFORMERS_CACHE"],
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "huggingface_token_present": hf_env["huggingface_token_present"],
        },
        "manifests": {
            "train": repo_relative_path(train_manifest),
            "val": repo_relative_path(val_manifest),
            "tests": {split_name: repo_relative_path(path) for split_name, path in test_manifests.items()},
        },
        "manifest_summary": manifest_summary,
        "target_validation": target_validation,
        "dependencies": dependencies,
        "torch_cuda": torch_cuda,
        "transformers_supports_gemma4": transformers_supports_gemma4,
        "model_access_check": model_access_check,
        "model_name_or_path": model_name_or_path,
        "local_model_exists": is_local_model,
        "blocked_reasons": blocked_reasons,
        "ready_to_train": not blocked_reasons,
        "notes": [
            "This path expects a Gemma 4 image-text checkpoint plus TRL/PEFT/datasets in the active environment.",
            "The current repo smoke dataset is intentionally small and is for pipeline proving, not for a production-quality student.",
        ],
    }


def run_training(config: dict[str, Any], output_dir: Path) -> None:
    from datasets import Image, load_dataset
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    dataset_dir = resolve_repo_path(config["dataset_dir"])
    if dataset_dir is None:
        raise ValueError("Could not resolve dataset_dir.")
    ensure_hf_cache_env(output_dir)
    train_path = dataset_dir / str(config["train_manifest"])
    val_path = dataset_dir / str(config["val_manifest"])

    dataset_dict = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )
    dataset_dict = dataset_dict.cast_column("image", Image())

    model_init_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": not bool(config.get("allow_remote_model_download", False)),
    }

    if bool(config["quantization"].get("load_in_4bit", False)):
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config["quantization"]["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=getattr(torch, str(config["quantization"]["bnb_4bit_compute_dtype"])),
            bnb_4bit_use_double_quant=bool(config["quantization"]["bnb_4bit_use_double_quant"]),
        )
        model_init_kwargs["device_map"] = config["quantization"].get("device_map", "auto")
    elif torch.cuda.is_available():
        model_init_kwargs["dtype"] = torch.bfloat16

    processor_name_or_path = config.get("processor_name_or_path") or config["model_name_or_path"]
    processor = AutoProcessor.from_pretrained(processor_name_or_path, local_files_only=not bool(config.get("allow_remote_model_download", False)))
    model = AutoModelForImageTextToText.from_pretrained(config["model_name_or_path"], **model_init_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False

    base_adapter_dir = resolve_repo_path(config.get("base_adapter_dir"))
    if base_adapter_dir is not None:
        if not base_adapter_dir.exists():
            raise FileNotFoundError(f"base_adapter_dir not found: {base_adapter_dir}")
        model = PeftModel.from_pretrained(model, str(base_adapter_dir), is_trainable=True)
    elif bool(config["lora"].get("enabled", True)):
        if bool(config["quantization"].get("load_in_4bit", False)):
            model = prepare_model_for_kbit_training(model)
        target_modules = config["lora"]["target_modules"]
        lora_cfg = LoraConfig(
            r=int(config["lora"]["r"]),
            lora_alpha=int(config["lora"]["alpha"]),
            lora_dropout=float(config["lora"]["dropout"]),
            bias="none",
            target_modules=target_modules if isinstance(target_modules, list) else str(target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    training_cfg = dict(config["training"])
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
            learning_rate=float(training_cfg["learning_rate"]),
            warmup_ratio=float(training_cfg["warmup_ratio"]),
            num_train_epochs=float(training_cfg["num_train_epochs"]),
            logging_steps=int(training_cfg["logging_steps"]),
            save_steps=int(training_cfg["save_steps"]),
            eval_steps=int(training_cfg["eval_steps"]),
            max_grad_norm=float(training_cfg["max_grad_norm"]),
            weight_decay=float(training_cfg["weight_decay"]),
            seed=int(training_cfg["seed"]),
            report_to=[],
            run_name=str(config["experiment_name"]),
            optim=str(training_cfg.get("optim", "adamw_torch_fused")),
            lr_scheduler_type=str(training_cfg.get("lr_scheduler_type", "linear")),
            max_length=training_cfg.get("max_length"),
            dataset_text_field="text",
            remove_unused_columns=False,
            eval_strategy="steps",
            save_strategy="steps",
            do_train=True,
            do_eval=True,
            logging_first_step=bool(training_cfg.get("logging_first_step", True)),
            save_total_limit=training_cfg.get("save_total_limit"),
            bf16=bool(torch.cuda.is_available()),
            tf32=bool(torch.cuda.is_available()),
        ),
    )
    trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "final_adapter"))
    trainer.save_state()


def main() -> None:
    args = build_parser().parse_args()
    config, config_path = load_config(args.config)
    output_dir = resolve_repo_path(config["output_dir"])
    if output_dir is None:
        raise ValueError("Could not resolve output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = gather_preflight(config, config_path)
    write_json(output_dir / "resolved_train_config.json", config)
    write_json(output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, indent=2, ensure_ascii=False))

    if args.preflight_only or not preflight["ready_to_train"]:
        return
    run_training(config, output_dir=output_dir)


if __name__ == "__main__":
    main()
