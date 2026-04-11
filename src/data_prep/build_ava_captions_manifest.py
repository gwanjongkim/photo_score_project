from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTIONS_DIR = Path(
    os.environ.get("AVA_CAPTIONS_DIR", str(REPO_ROOT / "data" / "raw" / "ava_captions"))
)
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "raw" / "ava" / "images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "ava_captions"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build AVA caption manifests matched to local AVA image files.")
    parser.add_argument("--captions_dir", default=str(DEFAULT_CAPTIONS_DIR))
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def normalize_image_id(image_id: object) -> tuple[str, str]:
    filename = Path(str(image_id)).name.lower()
    stem = Path(filename).stem.lower()
    return filename, stem


def index_images(image_dir: Path) -> dict[str, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    index: dict[str, Path] = {}
    for image_path in sorted(image_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        filename = image_path.name.lower()
        stem = image_path.stem.lower()
        index.setdefault(filename, image_path)
        index.setdefault(stem, image_path)
    if not index:
        raise ValueError(f"No AVA images were found under {image_dir}")
    return index


def resolve_image_path(image_id: object, image_index: dict[str, Path]) -> Path | None:
    filename, stem = normalize_image_id(image_id)
    return image_index.get(filename) or image_index.get(stem)


def to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def build_manifest_for_split(
    split_name: str,
    captions_dir: Path,
    image_index: dict[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    source_csv = captions_dir / f"{split_name}.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"Caption CSV not found: {source_csv}")

    frame = pd.read_csv(source_csv)
    required_columns = {"image_id", "comment", "MOS"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{source_csv} is missing required columns: {', '.join(missing_columns)}")

    resolved_paths = frame["image_id"].apply(lambda value: resolve_image_path(value, image_index))
    missing_image_mask = resolved_paths.isna()

    output_frame = frame.loc[~missing_image_mask].copy()
    output_frame.insert(
        0,
        "image_path",
        resolved_paths.loc[~missing_image_mask].apply(lambda path: to_repo_relative(path, REPO_ROOT)),
    )
    output_frame["image_id"] = output_frame["image_id"].astype(str)
    output_frame["comment"] = output_frame["comment"].fillna("").astype(str).str.strip()
    output_frame["mos"] = pd.to_numeric(output_frame["MOS"], errors="coerce")
    output_frame["split"] = split_name

    invalid_mos_mask = output_frame["mos"].isna()
    output_frame = output_frame.loc[~invalid_mos_mask].copy()

    preferred_columns = [
        column
        for column in ["image_path", "image_id", "comment", "mos", "challenge_id", "split"]
        if column in output_frame.columns
    ]
    remaining_columns = [
        column
        for column in output_frame.columns
        if column not in preferred_columns and column != "MOS"
    ]
    output_frame = output_frame[preferred_columns + remaining_columns]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}.csv"
    output_frame.to_csv(output_path, index=False)

    return {
        "split": split_name,
        "source_csv": str(source_csv),
        "output_csv": str(output_path),
        "input_rows": int(len(frame)),
        "matched_rows": int(len(output_frame)),
        "missing_image_rows": int(missing_image_mask.sum()),
        "invalid_mos_rows": int(invalid_mos_mask.sum()),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    captions_dir = Path(args.captions_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]

    image_index = index_images(image_dir)
    summaries = [
        build_manifest_for_split(
            split_name=split_name,
            captions_dir=captions_dir,
            image_index=image_index,
            output_dir=output_dir,
        )
        for split_name in splits
    ]

    summary = {
        "captions_dir": str(captions_dir),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "indexed_images": len({str(path) for path in image_index.values()}),
        "splits": summaries,
    }
    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
