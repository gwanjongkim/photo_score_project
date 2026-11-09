from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.gemma_distill.common import resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a Gemma LoRA adapter into the base model and save a merged artifact.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hf_home")
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def ensure_hf_env(hf_home: str | None) -> None:
    if not hf_home:
        return
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_home) / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "transformers")
    os.environ["HF_MODULES_CACHE"] = str(Path(hf_home) / "modules")


def main() -> None:
    args = build_parser().parse_args()
    adapter_dir = resolve_repo_path(args.adapter_dir)
    output_dir = resolve_repo_path(args.output_dir)
    if adapter_dir is None or output_dir is None:
        raise ValueError("Could not resolve adapter_dir or output_dir.")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_hf_env(args.hf_home)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        dtype=dtype,
        local_files_only=bool(args.local_files_only),
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_dir))
    processor = AutoProcessor.from_pretrained(str(adapter_dir), local_files_only=True)
    processor.save_pretrained(str(output_dir))
    print(str(output_dir))


if __name__ == "__main__":
    main()
