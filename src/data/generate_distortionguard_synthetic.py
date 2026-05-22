# DistortionGuard-IQA v1 합성 왜곡 데이터셋을 생성하는 스크립트.
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, features

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None


DEFAULT_INPUT_CSV = "data/processed/techiqa_guard/train.csv"
DEFAULT_OUT_DIR = "data/processed/distortionguard_iqa_v1/synthetic_smoke"
DEFAULT_MANIFEST_OUT = "data/processed/distortionguard_iqa_v1/synthetic_smoke_manifest.csv"
DEFAULT_PAIRS_OUT = "data/processed/distortionguard_iqa_v1/synthetic_smoke_pairs.csv"
DEFAULT_IMAGE_COL = "image_path"
DEFAULT_MAX_IMAGES = 100
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 4
DEFAULT_JPEG_QUALITY_MIN = 15
DEFAULT_JPEG_QUALITY_MAX = 80
MAX_OUTPUT_SIDE = 2048
SEVERITIES = (1, 2, 3, 4, 5)

DISTORTION_TYPES = (
    "gaussian_blur",
    "motion_blur",
    "defocus_blur",
    "gaussian_noise",
    "jpeg_compression",
    "webp_compression",
    "downscale_upscale",
    "underexposure",
    "overexposure",
    "contrast_loss",
)
DISTORTION_TYPE_IDS = {name: idx for idx, name in enumerate(DISTORTION_TYPES)}

MANIFEST_COLUMNS = [
    "source_image_path",
    "distorted_image_path",
    "filename",
    "source_dataset",
    "distortion_type",
    "distortion_type_id",
    "severity",
    "severity_norm",
    "expected_quality_order",
    "source_width",
    "source_height",
    "output_width",
    "output_height",
    "generation_status",
    "error",
]

PAIR_COLUMNS = [
    "ref_image_path",
    "distorted_a_path",
    "distorted_b_path",
    "source_image_path",
    "distortion_type",
    "severity_a",
    "severity_b",
    "label",
]

RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
RESAMPLE_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


@dataclass(frozen=True)
class GenerationConfig:
    image_col: str
    out_dir: str
    input_csv_name: str
    seed: int
    jpeg_quality_min: int
    jpeg_quality_max: int
    distortion_types: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a smoke synthetic distortion dataset for DistortionGuard-IQA v1."
    )
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--image_col", default=DEFAULT_IMAGE_COL)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest_out", default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--pairs_out", default=DEFAULT_PAIRS_OUT)
    parser.add_argument("--max_images", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--jpeg_quality_min", type=int, default=DEFAULT_JPEG_QUALITY_MIN)
    parser.add_argument("--jpeg_quality_max", type=int, default=DEFAULT_JPEG_QUALITY_MAX)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_images <= 0:
        raise ValueError("--max_images must be positive.")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive.")
    if not (1 <= args.jpeg_quality_min <= 100):
        raise ValueError("--jpeg_quality_min must be in [1, 100].")
    if not (1 <= args.jpeg_quality_max <= 100):
        raise ValueError("--jpeg_quality_max must be in [1, 100].")
    if args.jpeg_quality_min > args.jpeg_quality_max:
        raise ValueError("--jpeg_quality_min must be <= --jpeg_quality_max.")


def _summary_path(manifest_out: Path) -> Path:
    stem = manifest_out.stem
    if stem.endswith("_manifest"):
        stem = stem[: -len("_manifest")]
    return manifest_out.with_name(f"{stem}_summary.json")


def _safe_filename(source_image_path: str, fallback_filename: str | None) -> str:
    if fallback_filename is None or str(fallback_filename).lower() == "nan":
        raw_name = Path(source_image_path).name
    else:
        raw_name = fallback_filename
    stem = Path(str(raw_name)).stem or "image"
    clean_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "image"
    digest = hashlib.sha1(source_image_path.encode("utf-8")).hexdigest()[:10]
    return f"{clean_stem}_{digest}.jpg"


def _resolve_source_dataset(row: dict[str, Any], input_csv_name: str) -> str:
    for column in ("source_dataset", "dataset"):
        value = row.get(column)
        if value is not None and str(value) and str(value).lower() != "nan":
            return str(value)
    return Path(input_csv_name).stem


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _cap_image_size(image: Image.Image) -> Image.Image:
    width, height = image.size
    max_side = max(width, height)
    if max_side <= MAX_OUTPUT_SIDE:
        return image
    scale = MAX_OUTPUT_SIDE / float(max_side)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, RESAMPLE_LANCZOS)


def _quality_for_severity(severity: int, quality_min: int, quality_max: int) -> int:
    ratio = (severity - 1) / 4.0
    quality = round(quality_max - ratio * (quality_max - quality_min))
    return int(np.clip(quality, 1, 100))


def _roundtrip_compression(image: Image.Image, fmt: str, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format=fmt, quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def _save_jpeg(image: Image.Image, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, subsampling=0)


def _seed_for(seed: int, source_image_path: str, distortion_type: str, severity: int) -> int:
    payload = f"{seed}|{source_image_path}|{distortion_type}|{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _pil_kernel_motion_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    kernel = [0.0] * (kernel_size * kernel_size)
    center_row = kernel_size // 2
    for col in range(kernel_size):
        kernel[center_row * kernel_size + col] = 1.0
    return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), kernel, scale=kernel_size))


def _cv2_motion_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    array = np.asarray(image)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / float(kernel_size)
    blurred = cv2.filter2D(array, -1, kernel)
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8), mode="RGB")


def _cv2_defocus_blur(image: Image.Image, radius: int) -> Image.Image:
    y_grid, x_grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (x_grid * x_grid + y_grid * y_grid) <= radius * radius
    kernel = mask.astype(np.float32)
    kernel /= float(kernel.sum())
    blurred = cv2.filter2D(np.asarray(image), -1, kernel)
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8), mode="RGB")


def _apply_gaussian_noise(
    image: Image.Image,
    source_image_path: str,
    distortion_type: str,
    severity: int,
    seed: int,
) -> Image.Image:
    std_by_severity = {1: 5.0, 2: 10.0, 3: 18.0, 4: 28.0, 5: 42.0}
    rng = np.random.default_rng(_seed_for(seed, source_image_path, distortion_type, severity))
    array = np.asarray(image).astype(np.float32)
    noisy = array + rng.normal(0.0, std_by_severity[severity], size=array.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


def _apply_downscale_upscale(image: Image.Image, severity: int) -> Image.Image:
    factors = {1: 0.75, 2: 0.55, 3: 0.40, 4: 0.28, 5: 0.18}
    width, height = image.size
    factor = factors[severity]
    small_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    small = image.resize(small_size, RESAMPLE_BICUBIC)
    return small.resize((width, height), RESAMPLE_BICUBIC)


def _apply_contrast_loss(image: Image.Image, severity: int) -> Image.Image:
    blend_by_severity = {1: 0.15, 2: 0.30, 3: 0.45, 4: 0.60, 5: 0.75}
    array = np.asarray(image).astype(np.float32)
    gray_mean = float(array.mean())
    degraded = array * (1.0 - blend_by_severity[severity]) + gray_mean * blend_by_severity[severity]
    return Image.fromarray(np.clip(degraded, 0, 255).astype(np.uint8), mode="RGB")


def _apply_distortion(
    image: Image.Image,
    source_image_path: str,
    distortion_type: str,
    severity: int,
    config: GenerationConfig,
) -> tuple[Image.Image, int]:
    if distortion_type == "gaussian_blur":
        radius_by_severity = {1: 0.6, 2: 1.2, 3: 2.0, 4: 3.0, 5: 4.5}
        return image.filter(ImageFilter.GaussianBlur(radius_by_severity[severity])), 95
    if distortion_type == "motion_blur":
        kernel_by_severity = {1: 3, 2: 5, 3: 7, 4: 11, 5: 15}
        kernel_size = kernel_by_severity[severity]
        if cv2 is not None:
            return _cv2_motion_blur(image, kernel_size), 95
        return _pil_kernel_motion_blur(image, kernel_size), 95
    if distortion_type == "defocus_blur":
        radius_by_severity = {1: 1, 2: 2, 3: 3, 4: 5, 5: 7}
        radius = radius_by_severity[severity]
        if cv2 is not None:
            return _cv2_defocus_blur(image, radius), 95
        return image.filter(ImageFilter.GaussianBlur(float(radius))), 95
    if distortion_type == "gaussian_noise":
        return _apply_gaussian_noise(image, source_image_path, distortion_type, severity, config.seed), 95
    if distortion_type == "jpeg_compression":
        quality = _quality_for_severity(
            severity, config.jpeg_quality_min, config.jpeg_quality_max
        )
        return _roundtrip_compression(image, "JPEG", quality), 95
    if distortion_type == "webp_compression":
        quality = _quality_for_severity(
            severity, config.jpeg_quality_min, config.jpeg_quality_max
        )
        return _roundtrip_compression(image, "WEBP", quality), 95
    if distortion_type == "downscale_upscale":
        return _apply_downscale_upscale(image, severity), 95
    if distortion_type == "underexposure":
        factors = {1: 0.85, 2: 0.70, 3: 0.55, 4: 0.40, 5: 0.25}
        return ImageEnhance.Brightness(image).enhance(factors[severity]), 95
    if distortion_type == "overexposure":
        factors = {1: 1.15, 2: 1.35, 3: 1.60, 4: 1.90, 5: 2.30}
        return ImageEnhance.Brightness(image).enhance(factors[severity]), 95
    if distortion_type == "contrast_loss":
        return _apply_contrast_loss(image, severity), 95
    raise ValueError(f"Unsupported distortion type: {distortion_type}")


def _failed_manifest_row(
    source_image_path: str,
    filename: str,
    source_dataset: str,
    error: str,
    distortion_type: str | None = None,
    severity: int | None = None,
) -> dict[str, Any]:
    return {
        "source_image_path": source_image_path,
        "distorted_image_path": "",
        "filename": filename,
        "source_dataset": source_dataset,
        "distortion_type": distortion_type or "",
        "distortion_type_id": DISTORTION_TYPE_IDS.get(distortion_type, "") if distortion_type else "",
        "severity": severity if severity is not None else "",
        "severity_norm": (severity / 5.0) if severity is not None else "",
        "expected_quality_order": "",
        "source_width": "",
        "source_height": "",
        "output_width": "",
        "output_height": "",
        "generation_status": "failed",
        "error": error,
    }


def _process_source_record(payload: tuple[int, dict[str, Any], GenerationConfig]) -> dict[str, Any]:
    row_index, row, config = payload
    raw_source = row.get(config.image_col)
    source_image_path = "" if raw_source is None else str(raw_source)
    fallback_filename = row.get("filename")
    filename = _safe_filename(source_image_path or f"row_{row_index}", fallback_filename)
    source_dataset = _resolve_source_dataset(row, config.input_csv_name)

    result: dict[str, Any] = {
        "manifest_rows": [],
        "pair_rows": [],
        "failed_images": [],
        "failed_distortions": [],
        "source_processed": 0,
    }

    if not source_image_path or source_image_path.lower() == "nan":
        error = f"missing image path in column {config.image_col}"
        result["manifest_rows"].append(
            _failed_manifest_row(source_image_path, filename, source_dataset, error)
        )
        result["failed_images"].append({"source_image_path": source_image_path, "error": error})
        return result

    source_path = Path(source_image_path)
    try:
        source_image = _load_rgb_image(source_path)
        source_width, source_height = source_image.size
        image = _cap_image_size(source_image)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["manifest_rows"].append(
            _failed_manifest_row(source_image_path, filename, source_dataset, error)
        )
        result["failed_images"].append({"source_image_path": source_image_path, "error": error})
        return result

    output_width, output_height = image.size
    generated_by_type: dict[str, dict[int, str]] = {}

    for distortion_type in config.distortion_types:
        generated_by_type[distortion_type] = {}
        for severity in SEVERITIES:
            out_path = (
                Path(config.out_dir)
                / distortion_type
                / f"s{severity}"
                / filename
            )
            try:
                distorted, save_quality = _apply_distortion(
                    image=image,
                    source_image_path=source_image_path,
                    distortion_type=distortion_type,
                    severity=severity,
                    config=config,
                )
                _save_jpeg(distorted, out_path, quality=save_quality)
                generated_by_type[distortion_type][severity] = str(out_path)
                result["manifest_rows"].append(
                    {
                        "source_image_path": source_image_path,
                        "distorted_image_path": str(out_path),
                        "filename": filename,
                        "source_dataset": source_dataset,
                        "distortion_type": distortion_type,
                        "distortion_type_id": DISTORTION_TYPE_IDS[distortion_type],
                        "severity": severity,
                        "severity_norm": severity / 5.0,
                        "expected_quality_order": 6 - severity,
                        "source_width": source_width,
                        "source_height": source_height,
                        "output_width": output_width,
                        "output_height": output_height,
                        "generation_status": "generated",
                        "error": "",
                    }
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result["manifest_rows"].append(
                    _failed_manifest_row(
                        source_image_path=source_image_path,
                        filename=filename,
                        source_dataset=source_dataset,
                        error=error,
                        distortion_type=distortion_type,
                        severity=severity,
                    )
                )
                result["failed_distortions"].append(
                    {
                        "source_image_path": source_image_path,
                        "distortion_type": distortion_type,
                        "severity": severity,
                        "error": error,
                    }
                )

    for distortion_type, severity_to_path in generated_by_type.items():
        for severity, distorted_path in sorted(severity_to_path.items()):
            result["pair_rows"].append(
                {
                    "ref_image_path": source_image_path,
                    "distorted_a_path": source_image_path,
                    "distorted_b_path": distorted_path,
                    "source_image_path": source_image_path,
                    "distortion_type": distortion_type,
                    "severity_a": 0,
                    "severity_b": severity,
                    "label": 1,
                }
            )
        sorted_severities = sorted(severity_to_path)
        for index, severity_a in enumerate(sorted_severities):
            for severity_b in sorted_severities[index + 1 :]:
                result["pair_rows"].append(
                    {
                        "ref_image_path": source_image_path,
                        "distorted_a_path": severity_to_path[severity_a],
                        "distorted_b_path": severity_to_path[severity_b],
                        "source_image_path": source_image_path,
                        "distortion_type": distortion_type,
                        "severity_a": severity_a,
                        "severity_b": severity_b,
                        "label": 1,
                    }
                )

    result["source_processed"] = 1
    return result


def _load_source_records(input_csv: Path, image_col: str, max_images: int) -> list[dict[str, Any]]:
    df = pd.read_csv(input_csv)
    if image_col not in df.columns:
        raise KeyError(f"Missing image column '{image_col}' in {input_csv}")
    return df.head(max_images).to_dict(orient="records")


def _run_generation(
    records: list[dict[str, Any]],
    config: GenerationConfig,
    num_workers: int,
) -> list[dict[str, Any]]:
    payloads = [(idx, row, config) for idx, row in enumerate(records)]
    if num_workers == 1:
        return [_process_source_record(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(_process_source_record, payloads))


def _build_summary(
    args: argparse.Namespace,
    summary_out: Path,
    records: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    failed_images: list[dict[str, Any]],
    failed_distortions: list[dict[str, Any]],
    distortion_types: tuple[str, ...],
    webp_supported: bool,
) -> dict[str, Any]:
    generated_rows = [
        row for row in manifest_rows if row.get("generation_status") == "generated"
    ]
    successful_source_images = {
        str(row["source_image_path"]) for row in generated_rows if row.get("source_image_path")
    }
    status_counts: dict[str, int] = {}
    by_distortion: dict[str, dict[str, int]] = {
        distortion_type: {f"s{severity}": 0 for severity in SEVERITIES}
        for distortion_type in distortion_types
    }

    for row in manifest_rows:
        status = str(row.get("generation_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "generated":
            continue
        distortion_type = str(row["distortion_type"])
        severity_key = f"s{int(row['severity'])}"
        by_distortion.setdefault(
            distortion_type, {f"s{severity}": 0 for severity in SEVERITIES}
        )
        by_distortion[distortion_type][severity_key] += 1

    distortion_types_generated = [
        distortion_type
        for distortion_type, severity_counts in by_distortion.items()
        if sum(severity_counts.values()) > 0
    ]
    skipped_distortion_types = [
        distortion_type for distortion_type in DISTORTION_TYPES if distortion_type not in distortion_types
    ]

    return {
        "input_csv": str(args.input_csv),
        "image_col": str(args.image_col),
        "out_dir": str(args.out_dir),
        "manifest_out": str(args.manifest_out),
        "pairs_out": str(args.pairs_out),
        "summary_out": str(summary_out),
        "max_images": int(args.max_images),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "jpeg_quality_min": int(args.jpeg_quality_min),
        "jpeg_quality_max": int(args.jpeg_quality_max),
        "max_output_side": int(MAX_OUTPUT_SIDE),
        "source_images_requested": int(len(records)),
        "source_images_processed": int(len(successful_source_images)),
        "source_images_successful": int(len(records) - len(failed_images)),
        "source_images_failed": int(len(failed_images)),
        "distorted_images_generated": int(len(generated_rows)),
        "pairs_generated": int(len(pair_rows)),
        "failed_distortion_attempts": int(len(failed_distortions)),
        "generation_status_counts": status_counts,
        "distortion_types_requested": list(DISTORTION_TYPES),
        "distortion_types_generated": distortion_types_generated,
        "distortion_types_skipped": skipped_distortion_types,
        "webp_supported": bool(webp_supported),
        "by_distortion_type_and_severity": by_distortion,
        "failed_images": failed_images,
        "failed_distortions": failed_distortions[:100],
    }


def _write_outputs(
    manifest_out: Path,
    pairs_out: Path,
    summary_out: Path,
    manifest_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS).to_csv(manifest_out, index=False)
    pd.DataFrame(pair_rows, columns=PAIR_COLUMNS).to_csv(pairs_out, index=False)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(summary: dict[str, Any]) -> None:
    print("DistortionGuard-IQA v1 synthetic dataset generation summary")
    print(f"source_images_requested={summary['source_images_requested']}")
    print(f"source_images_successful={summary['source_images_successful']}")
    print(f"source_images_failed={summary['source_images_failed']}")
    print(f"distorted_images_generated={summary['distorted_images_generated']}")
    print(f"pairs_generated={summary['pairs_generated']}")
    print(f"webp_supported={summary['webp_supported']}")
    if summary["distortion_types_skipped"]:
        print(
            "WARNING: skipped distortion types: "
            + ", ".join(summary["distortion_types_skipped"]),
            file=sys.stderr,
        )
    print("by_distortion_type_and_severity:")
    for distortion_type, counts in summary["by_distortion_type_and_severity"].items():
        severity_counts = " ".join(f"{key}={value}" for key, value in counts.items())
        print(f"  {distortion_type}: {severity_counts}")


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    input_csv = Path(args.input_csv)
    manifest_out = Path(args.manifest_out)
    pairs_out = Path(args.pairs_out)
    summary_out = _summary_path(manifest_out)

    webp_supported = bool(features.check("webp"))
    distortion_types = tuple(
        distortion_type
        for distortion_type in DISTORTION_TYPES
        if distortion_type != "webp_compression" or webp_supported
    )
    if not webp_supported:
        print(
            "WARNING: Pillow WebP support is unavailable; skipping webp_compression.",
            file=sys.stderr,
        )

    records = _load_source_records(input_csv, args.image_col, args.max_images)
    config = GenerationConfig(
        image_col=args.image_col,
        out_dir=args.out_dir,
        input_csv_name=str(input_csv),
        seed=args.seed,
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
        distortion_types=distortion_types,
    )

    results = _run_generation(records, config, args.num_workers)
    manifest_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    failed_images: list[dict[str, Any]] = []
    failed_distortions: list[dict[str, Any]] = []
    for result in results:
        manifest_rows.extend(result["manifest_rows"])
        pair_rows.extend(result["pair_rows"])
        failed_images.extend(result["failed_images"])
        failed_distortions.extend(result["failed_distortions"])

    summary = _build_summary(
        args=args,
        summary_out=summary_out,
        records=records,
        manifest_rows=manifest_rows,
        pair_rows=pair_rows,
        failed_images=failed_images,
        failed_distortions=failed_distortions,
        distortion_types=distortion_types,
        webp_supported=webp_supported,
    )
    _write_outputs(manifest_out, pairs_out, summary_out, manifest_rows, pair_rows, summary)
    _print_summary(summary)


if __name__ == "__main__":
    main()
