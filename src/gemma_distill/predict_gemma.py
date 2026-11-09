from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from src.gemma_distill.common import resolve_repo_path
from src.gemma_distill.evaluate_gemma import extract_json_payload, validate_teacher_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Gemma student predictions for a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter_dir")
    parser.add_argument("--base_only", action="store_true")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--hf_home")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser


def ensure_hf_env(hf_home: str | None) -> None:
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_home) / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "transformers")
        os.environ["HF_MODULES_CACHE"] = str(Path(hf_home) / "modules")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return
    for token_path in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        if token_path.exists():
            candidate = token_path.read_text(encoding="utf-8").strip()
            if candidate:
                os.environ["HF_TOKEN"] = candidate
                return


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message["role"])
        content = message["content"]
        if role == "system" and isinstance(content, list) and content and content[0].get("type") == "text":
            normalized.append({"role": role, "content": content[0]["text"]})
        else:
            normalized.append({"role": role, "content": content})
    return normalized


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    adapter_dir = resolve_repo_path(args.adapter_dir) if args.adapter_dir else None
    output_path = resolve_repo_path(args.output_jsonl)
    if manifest_path is None or output_path is None:
        raise ValueError("Could not resolve manifest, adapter_dir, or output_jsonl.")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not args.base_only and adapter_dir is None:
        raise ValueError("--adapter_dir is required unless --base_only is set.")
    if adapter_dir is not None and not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hf_home = args.hf_home
    if hf_home is None:
        hf_home = str(((adapter_dir.parent if adapter_dir is not None else output_path.parent) / "hf_cache").resolve(strict=False))
    ensure_hf_env(hf_home)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if not args.base_only:
        from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor_source = str(adapter_dir) if adapter_dir is not None else args.base_model
    processor = AutoProcessor.from_pretrained(processor_source, local_files_only=True)
    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        **model_kwargs,
    )
    if not args.load_in_4bit:
        model = model.to(device)
    if not args.base_only:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()

    rows = load_jsonl(manifest_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt_messages = normalize_messages(list(row["messages"])[:2])
            prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            try:
                with Image.open(str(row["image"])) as image:
                    image_rgb = image.convert("RGB")
                inputs = processor(text=prompt, images=image_rgb, return_tensors="pt")
                inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=max(1, int(args.max_new_tokens)), do_sample=False)
                generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
                raw_prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()
                prediction_json, parse_error = extract_json_payload(raw_prediction)
                validation = validate_teacher_json(prediction_json, row.get("selection_state"))
                prediction = json.dumps(prediction_json, ensure_ascii=False) if prediction_json is not None else raw_prediction
                payload = {
                    "id": row["id"],
                    "split": row["split"],
                    "image_path": row["image_path"],
                    "prediction": prediction,
                    "raw_prediction": raw_prediction,
                    "prediction_json": prediction_json,
                    "parse_error": parse_error,
                    "format_valid": validation["is_valid"],
                    "validation_flags": validation["flags"],
                }
            except Exception as exc:
                payload = {
                    "id": row["id"],
                    "split": row["split"],
                    "image_path": row["image_path"],
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "format_valid": False,
                    "validation_flags": ["prediction_error"],
                }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(json.dumps({"id": row["id"], "format_valid": payload["format_valid"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
