from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.vila.model_loader import DEFAULT_MODEL_NAME, load_vision_language_model
from src.vila.prompt_sets import DEFAULT_PROMPT_PRESET, get_prompt_preset
from src.vila.score_with_prompts import PromptBasedVILAScorer, flatten_vila_result, load_image_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a folder of images with the VILA-lite prompt scorer.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt_preset", default=DEFAULT_PROMPT_PRESET)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.webp,.bmp")
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def iter_image_paths(input_dir: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    image_paths = []
    for path in input_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() in extensions:
            image_paths.append(path)
    return sorted(image_paths)


def write_outputs(
    results: list[dict[str, object]],
    output_dir: Path,
    prompt_preset: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "vila_scores.jsonl"
    output_csv = output_dir / "vila_scores.csv"

    with output_jsonl.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    fieldnames = [
        "image_path",
        "vila_score",
        "model_name",
        "model_backend",
        "prompt_preset",
        "prompt_good_image",
        "prompt_good_composition",
        "prompt_good_lighting",
        "prompt_clear_subject",
        "prompt_clean_background",
        "vila_explanation",
        "explanation_signals",
    ]
    rows = [flatten_vila_result(result, prompt_preset=prompt_preset) for result in results]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_jsonl, output_csv


def score_folder(
    input_dir: Path,
    recursive: bool,
    extensions: set[str],
    scorer: PromptBasedVILAScorer,
    batch_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    image_paths = iter_image_paths(input_dir=input_dir, recursive=recursive, extensions=extensions)
    if not image_paths:
        raise ValueError(f"No images found under {input_dir}")

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for start in range(0, len(image_paths), max(1, batch_size)):
        batch_paths = image_paths[start : start + max(1, batch_size)]
        openable_paths = []
        images = []
        for image_path in batch_paths:
            try:
                images.append(load_image_rgb(image_path))
                openable_paths.append(image_path)
            except Exception as exc:  # pragma: no cover - smoke-tested through CLI
                failures.append(
                    {
                        "image_path": str(image_path),
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
        if openable_paths:
            results.extend(scorer.score_loaded_images(openable_paths, images))
    return results, failures


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    extensions = {part.strip().lower() for part in args.extensions.split(",") if part.strip()}
    get_prompt_preset(args.prompt_preset)

    model = load_vision_language_model(
        model_name=args.model_name,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    scorer = PromptBasedVILAScorer(model=model, prompt_preset=args.prompt_preset)
    results, failures = score_folder(
        input_dir=input_dir,
        recursive=args.recursive,
        extensions=extensions,
        scorer=scorer,
        batch_size=args.batch_size,
    )
    output_jsonl, output_csv = write_outputs(results, output_dir=output_dir, prompt_preset=args.prompt_preset)

    summary = {
        "input_dir": str(input_dir),
        "num_images": len(results),
        "num_failures": len(failures),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "model_name": args.model_name,
        "prompt_preset": args.prompt_preset,
        "failures": failures,
    }
    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
