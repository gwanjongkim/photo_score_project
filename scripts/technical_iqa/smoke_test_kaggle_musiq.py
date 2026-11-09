# 공식 Google MUSIQ Kaggle 모델을 5장 이미지로 스모크 테스트한다.
from __future__ import annotations

import csv
import importlib.metadata
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "musiq_kaggle_smoke_20260601"
PREDICTIONS_CSV = OUTPUT_DIR / "smoke_predictions.csv"
REPORT_MD = OUTPUT_DIR / "report.md"
TEST_SAMPLES_DIR = ROOT / "test_samples"
MANIFEST_CSV = (
    ROOT
    / "outputs"
    / "research_protocol_iqa_manifests_20260530"
    / "benchmark_all_research_protocol_iqa.csv"
)

VARIANTS = {
    "koniq-10k": "https://www.kaggle.com/models/google/musiq/TensorFlow2/koniq-10k/1",
    "spaq": "https://www.kaggle.com/models/google/musiq/TensorFlow2/spaq/1",
    "paq2piq": "https://www.kaggle.com/models/google/musiq/TensorFlow2/paq2piq/1",
}


@dataclass
class VariantStatus:
    name: str
    handle: str
    load_method: str
    loaded: bool
    error: str = ""
    output_scale_hint: str = "unknown"


def select_images() -> list[Path]:
    sample_images = sorted(
        path
        for path in TEST_SAMPLES_DIR.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(sample_images) >= 5:
        return sample_images[:5]

    manifest_images: list[Path] = []
    if MANIFEST_CSV.exists():
        with MANIFEST_CSV.open(newline="") as f:
            for row in csv.DictReader(f):
                path = Path(row["image_path"])
                if path.exists() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    manifest_images.append(path)
                if len(manifest_images) == 5:
                    break
    if len(manifest_images) >= 5:
        return manifest_images

    raise RuntimeError("Could not find 5 local smoke-test images.")


def load_image(path: Path) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(str(path))
    image_uint8 = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image_uint8 = tf.ensure_shape(image_uint8, [None, None, 3])
    image_float = tf.image.convert_image_dtype(image_uint8, tf.float32)
    return raw, image_uint8, image_float


def flatten_output(output: Any) -> list[float]:
    if isinstance(output, dict):
        values: list[float] = []
        for key in sorted(output):
            values.extend(flatten_output(output[key]))
        return values
    if isinstance(output, (tuple, list)):
        values = []
        for item in output:
            values.extend(flatten_output(item))
        return values
    tensor = tf.convert_to_tensor(output)
    return [float(value) for value in tf.reshape(tensor, [-1]).numpy().tolist()]


def call_model(
    model: Any,
    image_bytes: tf.Tensor,
    image_uint8: tf.Tensor,
    image_float: tf.Tensor,
) -> tuple[list[float], str]:
    batched_bytes = tf.expand_dims(image_bytes, 0)
    batched_uint8 = tf.expand_dims(image_uint8, 0)
    batched_float = tf.expand_dims(image_float, 0)
    candidates = [
        ("call_uint8_batched", lambda: model(batched_uint8)),
        ("call_float01_batched", lambda: model(batched_float)),
    ]

    signatures = getattr(model, "signatures", {})
    if signatures:
        signature = signatures.get("serving_default") or next(iter(signatures.values()))
        kwargs = signature.structured_input_signature[1]
        positional = signature.structured_input_signature[0]
        if kwargs:
            input_name = next(iter(kwargs))
            candidates.extend(
                [
                    (
                        f"signature_{input_name}_bytes_batched",
                        lambda input_name=input_name: signature(**{input_name: batched_bytes}),
                    ),
                    (
                        f"signature_{input_name}_bytes_scalar",
                        lambda input_name=input_name: signature(**{input_name: image_bytes}),
                    ),
                    (
                        f"signature_{input_name}_uint8_batched",
                        lambda input_name=input_name: signature(**{input_name: batched_uint8}),
                    ),
                    (
                        f"signature_{input_name}_float01_batched",
                        lambda input_name=input_name: signature(**{input_name: batched_float}),
                    ),
                ]
            )
        elif positional:
            candidates.extend(
                [
                    ("signature_uint8_batched", lambda: signature(batched_uint8)),
                    ("signature_float01_batched", lambda: signature(batched_float)),
                ]
            )

    errors = []
    for method, fn in candidates:
        try:
            values = flatten_output(fn())
            return values, method
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {exc}")

    raise RuntimeError("All inference attempts failed. " + " | ".join(errors))


def infer_scale(values: list[float]) -> str:
    if not values:
        return "unknown"
    low = min(values)
    high = max(values)
    if 0.0 <= low and high <= 1.0:
        return "likely 0..1"
    if 0.0 <= low and high <= 10.0:
        return "likely 0..10"
    if 0.0 <= low and high <= 100.0:
        return "likely 0..100"
    return "unknown"


def load_with_tfhub(handle: str) -> Any:
    return hub.load(handle)


def try_kagglehub_download(variant: str) -> tuple[str | None, str]:
    try:
        import kagglehub

        path = kagglehub.model_download(f"google/musiq/tensorFlow2/{variant}/1")
        return str(path), ""
    except Exception:
        return None, traceback.format_exc(limit=8)


def write_report(
    statuses: list[VariantStatus],
    image_paths: list[Path],
    rows: list[dict[str, str]],
    package_notes: list[str],
) -> str:
    loaded = [status for status in statuses if status.loaded]
    failed = [status for status in statuses if not status.loaded]

    if loaded:
        decision = "MUSIQ_READY_FOR_100_IMAGE_SANITY"
    elif any("Temporary failure in name resolution" in status.error for status in failed):
        decision = "MUSIQ_BLOCKED_BY_KAGGLE_ACCESS"
    else:
        decision = "MUSIQ_BLOCKED_BY_TF_HUB_LOADING"

    lines = [
        "# Official Kaggle MUSIQ Smoke Test",
        "",
        "## Scope",
        "- Smoke test only.",
        "- No training, TFLite export, Flutter edits, production scoring edits, or distillation.",
        "- Local repo-practical MUSIQ was not imported or used.",
        "",
        "## Packages",
        *[f"- {note}" for note in package_notes],
        "",
        "## Images",
        *[f"- {path}" for path in image_paths],
        "",
        "## Variant Status",
    ]

    for status in statuses:
        lines.extend(
            [
                f"### {status.name}",
                f"- handle: `{status.handle}`",
                f"- load_method: `{status.load_method}`",
                f"- loaded: `{status.loaded}`",
                f"- output_scale_hint: `{status.output_scale_hint}`",
            ]
        )
        if status.error:
            lines.extend(["- error:", "```text", status.error.strip(), "```"])

    lines.extend(["", "## Raw Outputs"])
    if rows:
        for row in rows:
            lines.append(
                f"- {row['variant']} | {Path(row['image_path']).name} | "
                f"{row['call_method']} | {row['raw_outputs']}"
            )
    else:
        lines.append("- No raw outputs; no variant loaded successfully.")

    lines.extend(["", "## Final Decision", decision, ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    return decision


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = select_images()
    package_notes = [
        f"tensorflow: {tf.__version__}",
        f"tensorflow-hub: {hub.__version__}",
        f"kagglehub: {importlib.metadata.version('kagglehub')}",
        f"setuptools: {importlib.metadata.version('setuptools')} "
        "(pinned below 81 so tensorflow-hub can import pkg_resources)",
        "kagglehub fallback was not needed when tensorflow_hub direct loading succeeded",
    ]

    rows: list[dict[str, str]] = []
    statuses: list[VariantStatus] = []

    for variant, handle in VARIANTS.items():
        print(f"\n=== Loading {variant}: {handle}")
        status = VariantStatus(
            name=variant,
            handle=handle,
            load_method="tensorflow_hub_direct",
            loaded=False,
        )
        try:
            model = load_with_tfhub(handle)
        except Exception:
            direct_error = traceback.format_exc(limit=8)
            local_path, kaggle_error = try_kagglehub_download(variant)
            if local_path:
                status.load_method = "kagglehub_download_then_tensorflow_hub"
                status.handle = local_path
                try:
                    model = load_with_tfhub(local_path)
                except Exception:
                    status.error = (
                        "Direct tensorflow_hub load failed:\n"
                        + direct_error
                        + "\nKagglehub download succeeded, but local tensorflow_hub load failed:\n"
                        + traceback.format_exc(limit=8)
                    )
                    statuses.append(status)
                    print(status.error)
                    continue
            else:
                status.error = (
                    "Direct tensorflow_hub load failed:\n"
                    + direct_error
                    + "\nKagglehub fallback failed:\n"
                    + kaggle_error
                )
                statuses.append(status)
                print(status.error)
                continue

        variant_values: list[float] = []
        try:
            for image_path in image_paths:
                image_bytes, image_uint8, image_float = load_image(image_path)
                values, call_method = call_model(model, image_bytes, image_uint8, image_float)
                variant_values.extend(values)
                raw_outputs = repr(values)
                rows.append(
                    {
                        "variant": variant,
                        "load_method": status.load_method,
                        "call_method": call_method,
                        "image_path": str(image_path),
                        "raw_outputs": raw_outputs,
                        "score_scale_hint": infer_scale(values),
                    }
                )
                print(f"{variant} | {image_path.name} | {call_method} | {raw_outputs}")
            status.loaded = True
            status.output_scale_hint = infer_scale(variant_values)
        except Exception:
            status.error = "Model loaded, but inference failed:\n" + traceback.format_exc(limit=8)
            print(status.error)
        statuses.append(status)

    with PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant",
            "load_method",
            "call_method",
            "image_path",
            "raw_outputs",
            "score_scale_hint",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    decision = write_report(statuses, image_paths, rows, package_notes)
    print(f"\nWrote {PREDICTIONS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Final decision: {decision}")


if __name__ == "__main__":
    main()
