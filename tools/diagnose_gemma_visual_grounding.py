from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.gemma_distill.common import repo_relative_path, resolve_repo_path, write_json
from src.gemma_distill.evaluate_gemma import extract_json_payload, validate_teacher_json
from src.gemma_distill.predict_gemma import ensure_hf_env, normalize_messages


OBJECT_TERMS = {
    "bird",
    "birds",
    "boat",
    "building",
    "car",
    "chain",
    "child",
    "children",
    "city",
    "clock",
    "cloud",
    "dog",
    "face",
    "feather",
    "flower",
    "goose",
    "landscape",
    "leaf",
    "lion",
    "mountain",
    "ocean",
    "person",
    "portrait",
    "sky",
    "street",
    "tree",
    "water",
    "wheel",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe whether the Gemma adapter changes output when the image changes.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--base_model", default="google/gemma-4-E4B-it")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hf_home")
    parser.add_argument("--max_examples", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_examples(rows: list[dict[str, Any]], max_examples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_state: dict[str, list[dict[str, Any]]] = {"selected": [], "near_miss": [], "not_selected": []}
    for row in rows:
        state = str(row.get("selection_state") or "")
        if state in by_state:
            by_state[state].append(row)
    picked: list[dict[str, Any]] = []
    while len(picked) < max_examples and any(by_state.values()):
        for state in ("selected", "near_miss", "not_selected"):
            if len(picked) >= max_examples:
                break
            group = by_state[state]
            if not group:
                continue
            row = rng.choice(group)
            group.remove(row)
            picked.append(row)
    return picked


def prompt_for_row(processor: Any, row: dict[str, Any], *, include_image_marker: bool = True) -> str:
    messages = normalize_messages(list(row["messages"])[:2])
    if not include_image_marker:
        for message in messages:
            if message.get("role") != "user" or not isinstance(message.get("content"), list):
                continue
            message["content"] = [
                content
                for content in message["content"]
                if not (isinstance(content, dict) and content.get("type") == "image")
            ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_image(path: str | Path) -> Image.Image:
    with Image.open(str(path)) as image:
        return image.convert("RGB")


def make_blank(reference: Image.Image, output_path: Path) -> Image.Image:
    image = Image.new("RGB", reference.size, color=(0, 0, 0))
    image.save(output_path, quality=90)
    return image


def make_noise(reference: Image.Image, output_path: Path, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(reference.height, reference.width, 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    image.save(output_path, quality=90)
    return image


def text_for_similarity(payload: dict[str, Any] | None, fallback: str) -> str:
    if not isinstance(payload, dict):
        return fallback
    return " ".join(str(payload.get(key) or "") for key in ("comment_type", "short_reason", "detailed_reason", "comparison_reason"))


def object_hits(text: str) -> list[str]:
    lowered = text.lower()
    return sorted(term for term in OBJECT_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered))


def generate_one(
    *,
    model: Any,
    processor: Any,
    torch_module: Any,
    device: str,
    prompt: str,
    image: Image.Image | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = image
    inputs = processor(**kwargs)
    input_summary = {
        key: list(value.shape) if hasattr(value, "shape") else str(type(value))
        for key, value in inputs.items()
    }
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch_module.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max(1, int(max_new_tokens)), do_sample=False)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    raw = processor.decode(generated_ids, skip_special_tokens=True).strip()
    payload, parse_error = extract_json_payload(raw)
    return {
        "raw_prediction": raw,
        "prediction_json": payload,
        "parse_error": parse_error,
        "validation": validate_teacher_json(payload),
        "processor_input_summary": input_summary,
        "object_terms": object_hits(text_for_similarity(payload, raw)),
    }


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    adapter_dir = resolve_repo_path(args.adapter_dir)
    output_dir = resolve_repo_path(args.output_dir)
    if manifest_path is None or adapter_dir is None or output_dir is None:
        raise ValueError("Could not resolve manifest, adapter_dir, or output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    variants_dir = output_dir / "variant_images"
    variants_dir.mkdir(parents=True, exist_ok=True)

    hf_home = args.hf_home or str((adapter_dir.parent / "hf_cache").resolve(strict=False))
    ensure_hf_env(hf_home)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(str(adapter_dir), local_files_only=True)
    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)
    if not args.load_in_4bit:
        model = model.to(device)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()

    rows = pick_examples(load_jsonl(manifest_path), int(args.max_examples), int(args.seed))
    all_manifest_rows = load_jsonl(manifest_path)
    results_path = output_dir / "diagnostic_results.jsonl"
    summary_examples: list[dict[str, Any]] = []

    with results_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            original_path = Path(str(row["image"]))
            swapped_row = all_manifest_rows[(all_manifest_rows.index(row) + 1) % len(all_manifest_rows)]
            swapped_path = Path(str(swapped_row["image"]))
            original = load_image(original_path)
            swapped = load_image(swapped_path)
            blank_path = variants_dir / f"{index:02d}_{row['selection_state']}_blank.jpg"
            noise_path = variants_dir / f"{index:02d}_{row['selection_state']}_noise.jpg"
            blank = make_blank(original, blank_path)
            noise = make_noise(original, noise_path, seed=int(args.seed) + index)

            prompt_with_image = prompt_for_row(processor, row, include_image_marker=True)
            prompt_text_only = prompt_for_row(processor, row, include_image_marker=False)
            variants = [
                ("original", original, original_path, prompt_with_image),
                ("swapped_image_same_prompt", swapped, swapped_path, prompt_with_image),
                ("blank_black_same_prompt", blank, blank_path, prompt_with_image),
                ("noise_same_prompt", noise, noise_path, prompt_with_image),
                ("text_only_no_image_marker", None, None, prompt_text_only),
            ]
            row_results: list[dict[str, Any]] = []
            original_text = ""
            for variant_name, image, image_path, prompt in variants:
                try:
                    result = generate_one(
                        model=model,
                        processor=processor,
                        torch_module=torch,
                        device=device,
                        prompt=prompt,
                        image=image,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                except Exception as exc:
                    result = {
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "raw_prediction": "",
                        "prediction_json": None,
                        "parse_error": "generation_error",
                        "validation": {"is_valid": False, "flags": ["generation_error"]},
                        "processor_input_summary": {},
                        "object_terms": [],
                    }
                text = text_for_similarity(result.get("prediction_json"), str(result.get("raw_prediction") or ""))
                if variant_name == "original":
                    original_text = text
                similarity_to_original = difflib.SequenceMatcher(a=original_text, b=text).ratio() if original_text else 1.0
                payload = {
                    "example_index": index,
                    "id": row.get("id"),
                    "selection_state": row.get("selection_state"),
                    "selection_band": row.get("selection_band"),
                    "variant": variant_name,
                    "source_image": str(image_path) if image_path is not None else None,
                    "source_image_repo_relative": repo_relative_path(image_path) if image_path is not None and image_path.is_absolute() else str(image_path),
                    "prompt": prompt,
                    "similarity_to_original_variant": round(similarity_to_original, 6),
                    **result,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                row_results.append(payload)
            summary_examples.append(
                {
                    "id": row.get("id"),
                    "selection_state": row.get("selection_state"),
                    "variant_similarities": {
                        item["variant"]: item["similarity_to_original_variant"]
                        for item in row_results
                    },
                    "variant_object_terms": {
                        item["variant"]: item.get("object_terms", [])
                        for item in row_results
                    },
                    "variant_errors": {
                        item["variant"]: item.get("error")
                        for item in row_results
                        if item.get("error")
                    },
                    "predictions": {
                        item["variant"]: item.get("prediction_json")
                        for item in row_results
                    },
                }
            )

    variant_similarities: dict[str, list[float]] = {}
    text_only_errors = 0
    rows_written = 0
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows_written += 1
            item = json.loads(line)
            variant_similarities.setdefault(item["variant"], []).append(float(item["similarity_to_original_variant"]))
            if item["variant"] == "text_only_no_image_marker" and item.get("error"):
                text_only_errors += 1

    summary = {
        "manifest": repo_relative_path(manifest_path),
        "adapter_dir": repo_relative_path(adapter_dir),
        "base_model": args.base_model,
        "output_dir": repo_relative_path(output_dir),
        "examples": len(rows),
        "rows_written": rows_written,
        "device": device,
        "results_jsonl": repo_relative_path(results_path),
        "variant_similarity_to_original_mean": {
            variant: round(sum(values) / len(values), 6) if values else None
            for variant, values in sorted(variant_similarities.items())
        },
        "text_only_error_count": text_only_errors,
        "summary_examples": summary_examples,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
