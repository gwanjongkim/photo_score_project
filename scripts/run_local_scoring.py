#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AESTHETIC_CONFIG = REPO_ROOT / "configs" / "stage5_reference.json"
PATH_LIKE_BUNDLE_KEYS = {
    "aadb_model",
    "koniq_model",
    "flive_image_model",
    "flive_patch_model",
    "nima_model",
    "alamp_model",
    "musiq_model",
    "rgnet_model",
    "pairwise_model",
    "pairwise_reference_csv",
}


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def load_bundle_args(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Aesthetic config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Aesthetic config root must be an object: {config_path}")
    bundle_args = config.get("bundle_args")
    if not isinstance(bundle_args, dict):
        raise ValueError(f"Aesthetic config must contain object key 'bundle_args': {config_path}")
    return dict(bundle_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mode-1 local score calculation with stage5 model config.")
    parser.add_argument("--input_dir", default=os.environ.get("LOCAL_INPUT_DIR"))
    parser.add_argument("--output_dir", default=os.environ.get("LOCAL_OUTPUT_DIR"))
    parser.add_argument(
        "--aesthetic_config",
        default=os.environ.get("ACUT_AESTHETIC_CONFIG", str(DEFAULT_AESTHETIC_CONFIG)),
        help="Stage5 config path containing bundle_args model file locations.",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.webp,.bmp")
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.input_dir or not args.output_dir:
        raise ValueError(
            "Both --input_dir and --output_dir are required "
            "(or set LOCAL_INPUT_DIR and LOCAL_OUTPUT_DIR environment variables)."
        )
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    aesthetic_config = resolve_project_path(args.aesthetic_config)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    bundle_args = load_bundle_args(aesthetic_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "scores.jsonl"
    output_csv = output_dir / "scores.csv"

    command = [
        sys.executable,
        "-m",
        "src.infer.score_image_folder",
        "--input_dir",
        str(input_dir),
        "--output_jsonl",
        str(output_jsonl),
        "--output_csv",
        str(output_csv),
        "--extensions",
        str(args.extensions),
        "--json_indent",
        str(args.json_indent),
    ]
    if args.recursive:
        command.append("--recursive")

    for key in sorted(bundle_args.keys()):
        value = bundle_args[key]
        if value is None:
            continue

        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
            continue

        if key in PATH_LIKE_BUNDLE_KEYS:
            resolved = resolve_project_path(str(value))
            command.extend([f"--{key}", str(resolved)])
        else:
            command.extend([f"--{key}", str(value)])

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing_pythonpath else f"{REPO_ROOT}:{existing_pythonpath}"

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Local score run failed.\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    try:
        inner_summary = json.loads(result.stdout.strip()) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        inner_summary = {
            "raw_stdout": result.stdout.strip(),
        }

    summary = {
        "mode": "minimal_local_scores",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "aesthetic_config": str(aesthetic_config),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "command": command,
        "inner_summary": inner_summary,
    }
    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
