from __future__ import annotations

import argparse
import json
import logging
import mimetypes
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tensorflow as tf

from src.datasets.native_size_dataset import prepare_alamp_inputs
from src.models.alamp import LayoutCueAugmentation, WeightedPatchPooling
from src.models.nima_distribution import distribution_mean_score, emd_loss, mean_score_mae
from src.models.pairwise_comparator import AbsoluteDifference
from src.models.rgnet import GraphConvolution, RegionGraphBuilder, RegionWeightedPooling
from src.utils.pairwise_utils import pairwise_probabilities_to_strength, recover_score_from_pairwise_matrix


class BundleInferenceError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        image_path: str | Path,
        message: str,
        recoverable: bool,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.image_path = str(image_path)
        self.recoverable = bool(recoverable)


logger = logging.getLogger(__name__)


_HEIF_BRANDS = {
    "heic",
    "heix",
    "hevc",
    "hevx",
    "heim",
    "heis",
    "mif1",
    "msf1",
}


def _register_optional_heif_support() -> tuple[bool, str | None]:
    try:
        from pillow_heif import __version__ as pillow_heif_version
        from pillow_heif import register_heif_opener

        register_heif_opener()
        return True, f"pillow_heif {pillow_heif_version}"
    except Exception as exc:  # pragma: no cover - depends on worker environment
        return False, f"{exc.__class__.__name__}: {exc}"


HEIF_SUPPORT_ENABLED, HEIF_SUPPORT_DETAIL = _register_optional_heif_support()


def read_file_header_bytes(path: Path, num_bytes: int = 64) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(max(0, int(num_bytes)))
    except Exception:
        return b""


def detect_magic_format(header: bytes) -> str | None:
    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12].decode("ascii", errors="replace").lower()
        if brand in _HEIF_BRANDS or brand.startswith("he"):
            return f"image/heif({brand})"
        return f"application/isobmff({brand})"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith(b"BM"):
        return "image/bmp"
    return None


def header_hex_preview(header: bytes, max_bytes: int = 24) -> str:
    if not header:
        return ""
    return header[: max(0, int(max_bytes))].hex()


def load_image_rgb(image_path: Path) -> Image.Image:
    image_path = Path(image_path)
    try:
        with Image.open(image_path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
            image.load()
        width, height = image.size
        if width < 2 or height < 2:
            raise ValueError(f"Image dimensions are too small: {width}x{height}")
        return image
    except BundleInferenceError:
        raise
    except Exception as exc:
        header = read_file_header_bytes(image_path, num_bytes=64)
        magic_format = detect_magic_format(header)
        mime_guess, _ = mimetypes.guess_type(str(image_path))
        size_bytes = None
        try:
            size_bytes = int(image_path.stat().st_size)
        except OSError:
            pass
        details = (
            f"suffix={image_path.suffix.lower()!r}, "
            f"size_bytes={size_bytes}, "
            f"mime_guess={mime_guess!r}, "
            f"magic_format={magic_format!r}, "
            f"header_hex={header_hex_preview(header)!r}, "
            f"heif_support_enabled={HEIF_SUPPORT_ENABLED}, "
            f"heif_support_detail={HEIF_SUPPORT_DETAIL!r}"
        )
        hint = ""
        if magic_format and magic_format.startswith("image/heif") and not HEIF_SUPPORT_ENABLED:
            hint = " HEIF bytes detected but HEIF decoder support is unavailable in this environment."
        raise BundleInferenceError(
            stage="image_decode",
            image_path=image_path,
            message=f"{exc.__class__.__name__}: {exc}. [{details}]{hint}",
            recoverable=False,
        ) from exc


def preprocess_image_pil(img: Image.Image, size: int = 224) -> np.ndarray:
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def compute_patch_boxes(width: int, height: int, patch_size_ratio: float = 0.5) -> list[tuple[int, int, int, int]]:
    short_side = min(width, height)
    crop_size = max(32, int(short_side * patch_size_ratio))
    crop_size = min(crop_size, width, height)

    positions = [
        (0.5, 0.5),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
    ]

    boxes = []
    for px, py in positions:
        cx = int(width * px)
        cy = int(height * py)
        left = max(0, min(width - crop_size, cx - crop_size // 2))
        top = max(0, min(height - crop_size, cy - crop_size // 2))
        right = left + crop_size
        bottom = top + crop_size
        boxes.append((left, top, right, bottom))
    return boxes


def make_patch_batch(img: Image.Image, size: int = 224, patch_size_ratio: float = 0.5) -> np.ndarray:
    boxes = compute_patch_boxes(img.width, img.height, patch_size_ratio=patch_size_ratio)
    patches = []
    for box in boxes:
        patch = img.crop(box).resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(patch).astype("float32") / 255.0
        patches.append(arr)
    return np.stack(patches, axis=0)


def predict_single(model: tf.keras.Model, img_arr: np.ndarray) -> float:
    pred = model.predict(img_arr[None, ...], verbose=0)
    return float(np.squeeze(pred))


def predict_batch(model: tf.keras.Model, batch_arr: np.ndarray) -> np.ndarray:
    pred = model.predict(batch_arr, verbose=0)
    return np.squeeze(pred).astype("float32")


class SavedModelPredictAdapter:
    def __init__(self, saved_model_dir: str | Path, signature_name: str = "serving_default"):
        self.saved_model_dir = str(saved_model_dir)
        self.name = Path(saved_model_dir).name
        self._loaded = tf.saved_model.load(self.saved_model_dir)
        self._signature = self._loaded.signatures[signature_name]

    def predict(self, inputs, verbose: int = 0) -> np.ndarray:
        del verbose
        if not isinstance(inputs, dict):
            raise TypeError("SavedModelPredictAdapter expects dict inputs.")
        outputs = self._signature(
            **{key: tf.convert_to_tensor(value) for key, value in inputs.items()}
        )
        return next(iter(outputs.values())).numpy()


def load_model(path: str | None, custom_objects: dict | None = None, safe_mode: bool = True):
    if not path:
        return None
    return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects, safe_mode=safe_mode)


def load_musiq_model(path: str | None):
    if not path:
        return None
    try:
        return load_model(path, safe_mode=False)
    except (ValueError, NotImplementedError):
        saved_model_dir = Path(path).parent / "saved_model"
        if saved_model_dir.exists():
            return SavedModelPredictAdapter(saved_model_dir)
        raise


def _load_optional_bundle_model(args: argparse.Namespace, field_name: str, loader):
    try:
        return loader()
    except Exception as exc:
        auto_resolved_fields = set(getattr(args, "_auto_resolved_model_fields", []))
        if field_name not in auto_resolved_fields:
            raise
        warnings = list(getattr(args, "_model_load_warnings", []))
        warnings.append(
            {
                "field": field_name,
                "path": getattr(args, field_name, None),
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        args._model_load_warnings = warnings
        return None


def normalize_scores(aadb_score, koniq_score, flive_img_score, flive_patch_mean, flive_patch_min):
    aadb_norm = float(np.clip(aadb_score, 0.0, 1.0))
    koniq_norm = float(np.clip(koniq_score / 100.0, 0.0, 1.0))
    flive_img_norm = float(np.clip(flive_img_score / 100.0, 0.0, 1.0))
    flive_patch_mean_norm = float(np.clip(flive_patch_mean / 100.0, 0.0, 1.0))
    flive_patch_min_norm = float(np.clip(flive_patch_min / 100.0, 0.0, 1.0))
    return aadb_norm, koniq_norm, flive_img_norm, flive_patch_mean_norm, flive_patch_min_norm


def fuse_scores(aadb_score, koniq_score, flive_img_score, flive_patch_mean, flive_patch_min):
    aadb_norm, koniq_norm, flive_img_norm, flive_patch_mean_norm, flive_patch_min_norm = normalize_scores(
        aadb_score, koniq_score, flive_img_score, flive_patch_mean, flive_patch_min
    )

    base_score = (
        0.30 * aadb_norm
        + 0.30 * koniq_norm
        + 0.20 * flive_img_norm
        + 0.20 * flive_patch_mean_norm
    )

    penalty = max(0.0, 0.45 - flive_patch_min_norm) * 0.5
    final_score = max(0.0, base_score - penalty)

    return {
        "aadb_norm": aadb_norm,
        "koniq_norm": koniq_norm,
        "flive_img_norm": flive_img_norm,
        "flive_patch_mean_norm": flive_patch_mean_norm,
        "flive_patch_min_norm": flive_patch_min_norm,
        "penalty": penalty,
        "final_score": final_score,
    }


def parse_scale_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def normalized_boxes_to_pixels(boxes: np.ndarray, width: int, height: int) -> list[list[int]]:
    out = []
    for y1, x1, y2, x2 in boxes.tolist():
        out.append(
            [
                int(round(x1 * width)),
                int(round(y1 * height)),
                int(round(x2 * width)),
                int(round(y2 * height)),
            ]
        )
    return out


def build_musiq_inputs(
    img: Image.Image,
    patch_size: int,
    scale_sizes: tuple[int, ...],
    patches_per_scale: int,
) -> dict[str, np.ndarray]:
    patches_all = []
    positions_all = []
    scale_ids_all = []
    mask_all = []
    width, height = img.size

    for scale_idx, target_long in enumerate(scale_sizes):
        scale = float(target_long) / max(width, height)
        resized_w = max(patch_size, int(round(width * scale)))
        resized_h = max(patch_size, int(round(height * scale)))
        resized = np.asarray(img.resize((resized_w, resized_h), Image.Resampling.BILINEAR)).astype("float32") / 255.0
        patches = []
        positions = []
        for top in range(0, resized_h - patch_size + 1, patch_size):
            for left in range(0, resized_w - patch_size + 1, patch_size):
                patches.append(resized[top : top + patch_size, left : left + patch_size, :])
                positions.append([top // patch_size, left // patch_size])
        patches = patches[:patches_per_scale]
        positions = positions[:patches_per_scale]
        count = len(patches)
        while len(patches) < patches_per_scale:
            patches.append(np.zeros((patch_size, patch_size, 3), dtype="float32"))
            positions.append([0.0, 0.0])
        patches_all.extend(patches)
        positions_all.extend(positions)
        scale_ids_all.extend([scale_idx] * patches_per_scale)
        mask_all.extend([1.0] * count + [0.0] * (patches_per_scale - count))

    return {
        "patches": np.asarray(patches_all, dtype="float32")[None, ...],
        "positions": np.asarray(positions_all, dtype="float32")[None, ...],
        "scale_ids": np.asarray(scale_ids_all, dtype="int32")[None, ...],
        "token_mask": np.asarray(mask_all, dtype="float32")[None, ...],
    }


def add_bundle_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--aadb_model")
    parser.add_argument("--koniq_model")
    parser.add_argument("--flive_image_model")
    parser.add_argument("--flive_patch_model")
    parser.add_argument("--nima_model")
    parser.add_argument("--alamp_model")
    parser.add_argument("--musiq_model")
    parser.add_argument("--rgnet_model")
    parser.add_argument("--pairwise_model")
    parser.add_argument("--pairwise_reference_csv")
    parser.add_argument("--pairwise_target_col", default="score")
    parser.add_argument("--pairwise_max_references", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--rgnet_image_size", type=int, default=256)
    parser.add_argument("--alamp_global_size", type=int, default=384)
    parser.add_argument("--alamp_patch_size", type=int, default=224)
    parser.add_argument("--alamp_num_patches", type=int, default=5)
    parser.add_argument("--alamp_patch_scale", type=float, default=0.35)
    parser.add_argument("--alamp_include_debug", action="store_true")
    parser.add_argument("--include_debug", action="store_true")
    parser.add_argument("--musiq_patch_size", type=int, default=32)
    parser.add_argument("--musiq_scale_sizes", default="224,384,512")
    parser.add_argument("--musiq_patches_per_scale", type=int, default=16)
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run optional multi-model image quality and aesthetics bundle inference.")
    parser.add_argument("--image_path", required=True)
    return add_bundle_arguments(parser)


def load_bundle_models(args: argparse.Namespace) -> dict[str, tf.keras.Model | None]:
    return {
        "aadb": _load_optional_bundle_model(
            args,
            "aadb_model",
            lambda: load_model(args.aadb_model),
        ),
        "koniq": _load_optional_bundle_model(
            args,
            "koniq_model",
            lambda: load_model(args.koniq_model),
        ),
        "flive_image": _load_optional_bundle_model(
            args,
            "flive_image_model",
            lambda: load_model(args.flive_image_model),
        ),
        "flive_patch": _load_optional_bundle_model(
            args,
            "flive_patch_model",
            lambda: load_model(args.flive_patch_model),
        ),
        "nima": _load_optional_bundle_model(
            args,
            "nima_model",
            lambda: load_model(
                args.nima_model,
                custom_objects={"emd_loss": emd_loss, "mean_score_mae": mean_score_mae},
            ),
        ),
        "alamp": _load_optional_bundle_model(
            args,
            "alamp_model",
            lambda: load_model(
                args.alamp_model,
                custom_objects={
                    "LayoutCueAugmentation": LayoutCueAugmentation,
                    "WeightedPatchPooling": WeightedPatchPooling,
                },
                safe_mode=False,
            ),
        ),
        "musiq": _load_optional_bundle_model(
            args,
            "musiq_model",
            lambda: load_musiq_model(args.musiq_model),
        ),
        "rgnet": _load_optional_bundle_model(
            args,
            "rgnet_model",
            lambda: load_model(
                args.rgnet_model,
                custom_objects={
                    "RegionGraphBuilder": RegionGraphBuilder,
                    "GraphConvolution": GraphConvolution,
                    "RegionWeightedPooling": RegionWeightedPooling,
                },
                safe_mode=False,
            ),
        ),
        "pairwise": _load_optional_bundle_model(
            args,
            "pairwise_model",
            lambda: load_model(
                args.pairwise_model,
                custom_objects={"AbsoluteDifference": AbsoluteDifference},
                safe_mode=False,
            ),
        ),
    }


def build_summary(per_model: dict[str, dict], fused: dict | None) -> dict[str, object]:
    ordered_keys = [
        "baseline_aadb",
        "baseline_koniq",
        "baseline_flive_image",
        "baseline_flive_patch",
        "nima",
        "alamp",
        "musiq",
        "rgnet",
        "pairwise",
    ]
    available_models = [key for key in ordered_keys if key in per_model]
    aesthetic_models_present = [
        key
        for key in ["baseline_aadb", "nima", "alamp", "rgnet", "pairwise"]
        if key in per_model
    ]
    technical_models_present = [
        key
        for key in ["baseline_koniq", "baseline_flive_image", "baseline_flive_patch", "musiq"]
        if key in per_model
    ]
    return {
        "available_models": available_models,
        "aesthetic_models_present": aesthetic_models_present,
        "technical_models_present": technical_models_present,
        "reranking_ready": bool(aesthetic_models_present) and (bool(technical_models_present) or "pairwise" in per_model),
        "baseline_fusion_available": fused is not None,
    }


def add_legacy_fields(result: dict[str, object], per_model: dict[str, dict], fused: dict | None, debug: dict | None) -> None:
    if "baseline_aadb" in per_model:
        result["aadb_score"] = per_model["baseline_aadb"]["score"]
    if "baseline_koniq" in per_model:
        result["koniq_score"] = per_model["baseline_koniq"]["score"]
    if "baseline_flive_image" in per_model:
        result["flive_image_score"] = per_model["baseline_flive_image"]["score"]
    if "baseline_flive_patch" in per_model:
        result["flive_patch_mean"] = per_model["baseline_flive_patch"]["mean_score"]
        result["flive_patch_min"] = per_model["baseline_flive_patch"]["min_score"]
        result["flive_patch_std"] = per_model["baseline_flive_patch"]["std_score"]
        if debug and "baseline_flive_patch" in debug:
            result["flive_patch_scores"] = debug["baseline_flive_patch"]["patch_scores"]
    if fused is not None:
        result.update(fused)
    if "nima" in per_model:
        result["nima_mean_score"] = per_model["nima"]["mean_score"]
        if debug and "nima" in debug:
            result["nima_distribution"] = debug["nima"]["distribution"]
    if "alamp" in per_model:
        result["alamp_score"] = per_model["alamp"]["score"]
        if debug and "alamp" in debug:
            result["alamp_selected_patch_boxes"] = debug["alamp"]["selected_patch_boxes"]
            result["alamp_patch_selection_scores"] = debug["alamp"]["patch_selection_scores"]
            result["alamp_patch_attention_weights"] = debug["alamp"]["patch_attention_weights"]
    if "musiq" in per_model:
        result["musiq_score"] = per_model["musiq"]["score"]
    if "rgnet" in per_model:
        result["rgnet_score"] = per_model["rgnet"]["score"]
    if "pairwise" in per_model:
        result["pairwise_recovered_score"] = per_model["pairwise"]["recovered_score"]
        result["pairwise_num_references"] = per_model["pairwise"]["num_references"]
        result["pairwise_query_eigen_weight"] = per_model["pairwise"]["query_eigen_weight"]


def predict_bundle_for_image(
    image_path: str | Path,
    args: argparse.Namespace,
    models: dict[str, tf.keras.Model | None] | None = None,
) -> dict[str, object]:
    if models is None:
        models = load_bundle_models(args)

    image_path = Path(image_path)
    img = load_image_rgb(image_path)
    try:
        img_arr = preprocess_image_pil(img, size=args.image_size)
    except Exception as exc:
        raise BundleInferenceError(
            stage="image_preprocess",
            image_path=image_path,
            message=f"{exc.__class__.__name__}: {exc}",
            recoverable=False,
        ) from exc
    include_debug = bool(getattr(args, "include_debug", False) or getattr(args, "alamp_include_debug", False))

    per_model: dict[str, dict] = {}
    debug: dict[str, dict] = {}
    stage_failures: list[dict[str, object]] = []

    aadb_score = koniq_score = flive_image_score = None
    flive_patch_mean = flive_patch_min = None

    def record_stage_failure(stage: str, exc: Exception) -> None:
        failure: dict[str, object] = {
            "image_path": str(image_path),
            "stage": stage,
            "recoverable": True,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
        if isinstance(exc, BundleInferenceError):
            failure["source_stage"] = exc.stage
            failure["source_image_path"] = exc.image_path
            failure["source_recoverable"] = exc.recoverable
        stage_failures.append(failure)
        logger.warning(
            "Image stage failure (recoverable): image=%s stage=%s error=%s",
            image_path,
            stage,
            failure["error"],
        )

    if models["aadb"] is not None:
        try:
            aadb_score = predict_single(models["aadb"], img_arr)
            per_model["baseline_aadb"] = {"score": aadb_score}
        except Exception as exc:
            record_stage_failure("model_aadb", exc)

    if models["koniq"] is not None:
        try:
            koniq_score = predict_single(models["koniq"], img_arr)
            per_model["baseline_koniq"] = {"score": koniq_score}
        except Exception as exc:
            record_stage_failure("model_koniq", exc)

    if models["flive_image"] is not None:
        try:
            flive_image_score = predict_single(models["flive_image"], img_arr)
            per_model["baseline_flive_image"] = {"score": flive_image_score}
        except Exception as exc:
            record_stage_failure("model_flive_image", exc)

    if models["flive_patch"] is not None:
        try:
            patch_batch = make_patch_batch(img, size=args.image_size, patch_size_ratio=0.5)
            patch_scores = predict_batch(models["flive_patch"], patch_batch)
            flive_patch_mean = float(np.mean(patch_scores))
            flive_patch_min = float(np.min(patch_scores))
            per_model["baseline_flive_patch"] = {
                "mean_score": flive_patch_mean,
                "min_score": flive_patch_min,
                "std_score": float(np.std(patch_scores)),
                "num_patches": int(len(patch_scores)),
            }
            if include_debug:
                debug["baseline_flive_patch"] = {
                    "patch_scores": patch_scores.tolist(),
                }
        except Exception as exc:
            record_stage_failure("model_flive_patch", exc)

    fused = None
    if None not in (aadb_score, koniq_score, flive_image_score, flive_patch_mean, flive_patch_min):
        fused = fuse_scores(
            aadb_score=aadb_score,
            koniq_score=koniq_score,
            flive_img_score=flive_image_score,
            flive_patch_mean=flive_patch_mean,
            flive_patch_min=flive_patch_min,
        )

    if models["nima"] is not None:
        try:
            dist = np.squeeze(models["nima"].predict(img_arr[None, ...], verbose=0)).astype("float32")
            mean_score = float(distribution_mean_score(tf.convert_to_tensor(dist[None, :]))[0].numpy())
            per_model["nima"] = {"mean_score": mean_score}
            if include_debug:
                debug["nima"] = {"distribution": dist.tolist()}
        except Exception as exc:
            record_stage_failure("model_nima", exc)

    if models["alamp"] is not None:
        try:
            image_tensor = tf.convert_to_tensor(np.asarray(img).astype("float32") / 255.0, dtype=tf.float32)
            patch_scales = tuple(
                sorted(
                    {
                        max(0.18, args.alamp_patch_scale * 0.7),
                        args.alamp_patch_scale,
                        min(0.55, args.alamp_patch_scale * 1.35),
                    }
                )
            )
            global_view, patches, boxes, proposal_scores = prepare_alamp_inputs(
                image=image_tensor,
                global_size=args.alamp_global_size,
                patch_size=args.alamp_patch_size,
                num_patches=args.alamp_num_patches,
                patch_scales=patch_scales,
            )
            alamp_inputs = {
                "global_view": global_view[None, ...].numpy(),
                "patches": patches[None, ...].numpy(),
            }
            alamp_score = float(np.squeeze(models["alamp"].predict(alamp_inputs, verbose=0)))
            per_model["alamp"] = {
                "score": alamp_score,
                "global_size": args.alamp_global_size,
                "patch_size": args.alamp_patch_size,
                "num_patches": args.alamp_num_patches,
            }
            if include_debug:
                try:
                    attention_model = tf.keras.Model(
                        models["alamp"].inputs,
                        {
                            "score": models["alamp"].output,
                            "patch_attention": models["alamp"].get_layer("patch_attention").output,
                        },
                    )
                    debug_pred = attention_model.predict(alamp_inputs, verbose=0)
                    patch_attention = np.squeeze(debug_pred["patch_attention"], axis=(0, 2)).astype("float32")
                    pixel_boxes = normalized_boxes_to_pixels(boxes.numpy(), img.width, img.height)
                    selection_scores = proposal_scores.numpy().astype("float32").tolist()
                    debug["alamp"] = {
                        "selected_patch_boxes": pixel_boxes,
                        "patch_selection_scores": selection_scores,
                        "patch_attention_weights": patch_attention.tolist(),
                        "patch_debug": [
                            {
                                "box_xyxy": box,
                                "selection_score": float(selection),
                                "attention_weight": float(attn),
                            }
                            for box, selection, attn in zip(pixel_boxes, selection_scores, patch_attention.tolist())
                        ],
                    }
                except Exception as exc:
                    record_stage_failure("debug_alamp", exc)
        except Exception as exc:
            record_stage_failure("model_alamp", exc)

    if models["musiq"] is not None:
        try:
            scale_sizes = parse_scale_sizes(args.musiq_scale_sizes)
            musiq_inputs = build_musiq_inputs(
                img=img,
                patch_size=args.musiq_patch_size,
                scale_sizes=scale_sizes,
                patches_per_scale=args.musiq_patches_per_scale,
            )
            musiq_score = float(np.squeeze(models["musiq"].predict(musiq_inputs, verbose=0)))
            per_model["musiq"] = {
                "score": musiq_score,
                "patch_size": args.musiq_patch_size,
                "scale_sizes": list(scale_sizes),
                "patches_per_scale": args.musiq_patches_per_scale,
            }
        except Exception as exc:
            record_stage_failure("model_musiq", exc)

    if models["rgnet"] is not None:
        try:
            rgnet_arr = preprocess_image_pil(img, size=args.rgnet_image_size)
            per_model["rgnet"] = {
                "score": float(np.squeeze(models["rgnet"].predict(rgnet_arr[None, ...], verbose=0))),
                "image_size": args.rgnet_image_size,
            }
        except Exception as exc:
            record_stage_failure("model_rgnet", exc)

    if models["pairwise"] is not None and args.pairwise_reference_csv:
        try:
            reference_df = pd.read_csv(args.pairwise_reference_csv)
            required_columns = {"image_path", args.pairwise_target_col}
            if not required_columns.issubset(reference_df.columns):
                raise ValueError("Pairwise reference CSV is missing required columns.")

            reference_df = reference_df[["image_path", args.pairwise_target_col]].head(args.pairwise_max_references).copy()
            query_image = preprocess_image_pil(img, size=args.image_size)

            ref_images: list[np.ndarray] = []
            ref_scores: list[float] = []
            for ref_row in reference_df.to_dict(orient="records"):
                ref_path = Path(str(ref_row["image_path"]))
                try:
                    ref_rgb = load_image_rgb(ref_path)
                    ref_images.append(preprocess_image_pil(ref_rgb, size=args.image_size))
                    ref_scores.append(float(ref_row[args.pairwise_target_col]))
                except Exception as exc:
                    record_stage_failure("model_pairwise_reference_decode", exc)

            if len(ref_images) < 2:
                raise ValueError("Pairwise scoring requires at least two valid reference images.")

            query_strengths = []
            for ref_image in ref_images:
                probs = np.squeeze(
                    models["pairwise"].predict(
                        {"image_a": query_image[None, ...], "image_b": ref_image[None, ...]},
                        verbose=0,
                    )
                ).astype("float32")
                query_strengths.append(pairwise_probabilities_to_strength(probs))

            ref_matrix = np.ones((len(ref_images), len(ref_images)), dtype=np.float32)
            for i in range(len(ref_images)):
                for j in range(i + 1, len(ref_images)):
                    probs = np.squeeze(
                        models["pairwise"].predict(
                            {"image_a": ref_images[i][None, ...], "image_b": ref_images[j][None, ...]},
                            verbose=0,
                        )
                    ).astype("float32")
                    strength = pairwise_probabilities_to_strength(probs)
                    ref_matrix[i, j] = strength
                    ref_matrix[j, i] = 1.0 / max(strength, 1e-6)

            recovered, eigen_scores = recover_score_from_pairwise_matrix(
                query_vs_refs=np.asarray(query_strengths, dtype=np.float32),
                reference_scores=np.asarray(ref_scores, dtype=np.float32),
                reference_vs_reference=ref_matrix,
            )
            per_model["pairwise"] = {
                "recovered_score": float(recovered),
                "num_references": int(len(ref_images)),
                "query_eigen_weight": float(eigen_scores[0]),
                "target_col": args.pairwise_target_col,
            }
            if include_debug:
                debug["pairwise"] = {
                    "query_vs_reference_strengths": [float(x) for x in query_strengths],
                }
        except Exception as exc:
            record_stage_failure("model_pairwise", exc)

    result: dict[str, object] = {
        "image_path": str(image_path),
        "per_model": per_model,
        "summary": build_summary(per_model, fused),
    }
    if fused is not None:
        result["fused"] = fused
    if include_debug and debug:
        result["debug"] = debug
    if stage_failures:
        result["model_stage_failures"] = stage_failures

    add_legacy_fields(result, per_model, fused, debug if include_debug else None)
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    models = load_bundle_models(args)
    result = predict_bundle_for_image(args.image_path, args=args, models=models)
    print(json.dumps(result, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
