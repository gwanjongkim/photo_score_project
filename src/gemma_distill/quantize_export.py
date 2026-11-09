from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.gemma_distill.common import read_json, repo_relative_path, resolve_repo_path, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantization/export preflight for the Gemma 4 E4B student.")
    parser.add_argument("--train_output_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_output_dir = resolve_repo_path(args.train_output_dir)
    output_dir = resolve_repo_path(args.output_dir)
    if train_output_dir is None or output_dir is None:
        raise ValueError("Could not resolve train_output_dir or output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight_path = train_output_dir / "preflight.json"
    resolved_config_path = train_output_dir / "resolved_train_config.json"
    final_adapter_dir = train_output_dir / "final_adapter"
    merged_model_dir = output_dir / "merged_model"
    quantized_dir = output_dir / "quantized"

    payload: dict[str, Any] = {
        "train_output_dir": repo_relative_path(train_output_dir),
        "output_dir": repo_relative_path(output_dir),
        "export_mode": "adapter_bundle_only_unless_quantized_artifact_exists",
        "quantization_target": "mobile_candidate_not_completed",
        "status": "blocked",
        "quantized_export_completed": False,
        "blocked_reasons": [],
    }
    if preflight_path.exists():
        payload["train_preflight"] = read_json(preflight_path)
    if resolved_config_path.exists():
        payload["resolved_train_config"] = read_json(resolved_config_path)

    if not final_adapter_dir.exists():
        payload["blocked_reasons"].append("No fine-tuned adapter directory exists at train_output_dir/final_adapter.")
    if payload.get("train_preflight", {}).get("ready_to_train") is False:
        payload["blocked_reasons"].append("Training preflight did not pass, so there is nothing valid to quantize/export.")

    if final_adapter_dir.exists():
        payload["adapter_export_artifacts"] = [repo_relative_path(final_adapter_dir)]
    if merged_model_dir.exists():
        payload["merged_export_artifacts"] = [repo_relative_path(merged_model_dir)]
    if quantized_dir.exists():
        payload["status"] = "quantized_exported"
        payload["quantized_export_completed"] = True
        payload["quantized_export_artifacts"] = [repo_relative_path(quantized_dir)]
        payload["exported_artifacts"] = [
            item
            for item in (
                repo_relative_path(final_adapter_dir) if final_adapter_dir.exists() else None,
                repo_relative_path(merged_model_dir) if merged_model_dir.exists() else None,
                repo_relative_path(quantized_dir),
            )
            if item
        ]
    elif final_adapter_dir.exists():
        payload["status"] = "adapter_export_only"
        payload["blocked_reasons"].append("Fine-tuned adapter exists, but no quantized export artifact was produced in this retry.")
        payload["exported_artifacts"] = [
            item
            for item in (
                repo_relative_path(final_adapter_dir) if final_adapter_dir.exists() else None,
                repo_relative_path(merged_model_dir) if merged_model_dir.exists() else None,
            )
            if item
        ]
    write_json(output_dir / "export_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
