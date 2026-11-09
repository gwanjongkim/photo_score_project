# MDIQA 데이터셋별 checkpoint variant의 100장 sanity benchmark를 실행한다.
from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
import traceback
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MDIQA_REPO = ROOT / "MDIQA"
MANIFEST_CSV = (
    ROOT
    / "outputs"
    / "research_protocol_iqa_manifests_20260530"
    / "benchmark_all_research_protocol_iqa.csv"
)
CUDA_REPORT = ROOT / "outputs" / "mdiqa_cuda_diagnosis_20260601" / "report.md"
OUTPUT_DIR = ROOT / "outputs" / "mdiqa_variant_sanity_20260601"
VARIANT_MANIFEST_CSV = OUTPUT_DIR / "variant_sanity_manifest.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
METRICS_CSV = OUTPUT_DIR / "dataset_metrics.csv"
BEST_VARIANT_CSV = OUTPUT_DIR / "best_variant_by_dataset.csv"
FAILURES_CSV = OUTPUT_DIR / "failures.csv"
INVENTORY_CSV = OUTPUT_DIR / "checkpoint_inventory.csv"
REPORT_MD = OUTPUT_DIR / "report.md"

SEED = 20260601
SAMPLE_PER_DATASET = 100
BATCH_SIZE = 8
IMAGE_SIZE = 384
DATASETS = {
    "flive": "FLIVE",
    "koniq10k": "KonIQ-10k",
    "spaq": "SPAQ",
}
PRODUCTION_BASELINES = {
    "flive": 0.574,
    "koniq10k": 0.934,
    "spaq": 0.873,
}
REMOTE_VISIBLE_FILES = {
    "mdiqa_shared_5t_4a_clipfeat_localv4_mbv3_koniq_seed1.pth": "1QdqiGR1sdKddfkQduuhUONnF8k_Iem6e",
    "mdiqa_shared_5t_4a_mbv3_koniq_seed1.pth": "1AHd8qlMFO7RLwZoeYEP09SidxWExpS3n",
    "mdiqa_shared_5t_4a_mbv3_koniq_xd.pth": "1z4LbbDhc5ROvibMpdaV3BadTJlj1ayyN",
    "net_best.pth": "1veLWnwjoMDMRitAP02KalOaUvpBQ8vKS",
    "net_best_PARA_val_color.pth": "10skPYtS4-RYoaWEDwAemv8yM8QeEvOaP",
    "net_best_koniq10k_val_head0_head0.pth": "1eNKeLtOi4nsaLLeZYR26dr2_kuE8NlMq",
    "net_best_koniq10k_val_head1_head1.pth": "1Y3dG5eNblFdP_f5cf86TapDE-ammqacn",
    "net_best_koniq10k_val_head2_head2.pth": "1qnS513ilxOHHgUWZ3yOP4W2BXOSehumk",
    "net_best_koniq10k_val_head3_head3.pth": "1k2bIz3iS_xJBqz6EYAmu6W3QAsQa47_A",
    "net_best_koniq10k_val_head4_head4.pth": "1Bt5ZGmhIdj0WcjD7gHhAJ7ntzChro3nD",
    "net_freeze_backbone.pth": "1i0tZ6ptiaBIhvuL9Qt0Uri_00NoqGCGQ",
}

SHARED_CHECKPOINTS = {
    "technology_backbone": MDIQA_REPO / "checkpoints" / "net_freeze_backbone.pth",
    "technology_head0": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head0_head0.pth",
    "technology_head1": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head1_head1.pth",
    "technology_head2": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head2_head2.pth",
    "technology_head3": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head3_head3.pth",
    "technology_head4": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head4_head4.pth",
    "aesthetic_color": MDIQA_REPO / "checkpoints" / "net_best_PARA_val_color.pth",
}
VARIANT_FUSION_CANDIDATES = {
    "mdiqa_koniq": [
        MDIQA_REPO / "checkpoints" / "mdiqa_shared_5t_4a_mbv3_koniq_seed1.pth",
    ],
    "mdiqa_xd": [
        MDIQA_REPO / "checkpoints" / "mdiqa_shared_5t_4a_mbv3_koniq_xd.pth",
    ],
    "mdiqa_unified": [
        MDIQA_REPO / "checkpoints" / "net_best.pth",
    ],
}


@dataclass
class RuntimeInfo:
    torch_version: str
    torch_cuda: str
    cuda_available: bool
    device_count: int
    device_name: str
    device_used: str


@dataclass
class VariantSpec:
    variant: str
    fusion_path: Path
    local_score: bool = False
    semantic_model_name: str = "mobilenetv3_large_100.ra_in1k"
    semantic_feat: str = "without"


def configure_local_caches() -> None:
    cache_root = OUTPUT_DIR / "cache"
    for key, subdir in {
        "HF_HOME": "huggingface",
        "TRANSFORMERS_CACHE": "huggingface",
        "MPLCONFIGDIR": "matplotlib",
        "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": "xdg",
    }.items():
        path = cache_root / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))


def install_mdiqa_import_aliases() -> None:
    impl_base = MDIQA_REPO / "MDIQA"
    impl_pkg = types.ModuleType("local_training_20240727")
    impl_pkg.__path__ = [str(impl_base)]
    sys.modules.setdefault("local_training_20240727", impl_pkg)
    for alias in ["utils", "archs", "models", "data", "losses"]:
        package_path = impl_base / alias
        for prefix in ["MDIQA", "local_training_20240727"]:
            module_name = f"{prefix}.{alias}"
            module = types.ModuleType(module_name)
            module.__path__ = [str(package_path)]
            sys.modules.setdefault(module_name, module)


def force_timm_no_pretrained() -> None:
    import timm

    original_create_model = timm.create_model

    def create_model_no_pretrain(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = create_model_no_pretrain


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_status(path: Path) -> tuple[bool, int]:
    return path.exists() and path.stat().st_size > 1024, path.stat().st_size if path.exists() else 0


def resolve_variants() -> tuple[list[VariantSpec], list[str], list[str]]:
    variants: list[VariantSpec] = []
    missing_variants: list[str] = []
    missing_shared: list[str] = []
    for name, path in SHARED_CHECKPOINTS.items():
        present, _ = file_status(path)
        if not present:
            missing_shared.append(f"{name}: {path}")

    # Standard Koniq seed1
    p_koniq = MDIQA_REPO / "checkpoints" / "mdiqa_shared_5t_4a_mbv3_koniq_seed1.pth"
    if p_koniq.exists():
        variants.append(VariantSpec("mdiqa_koniq", p_koniq))

    # Cross-Dataset
    p_xd = MDIQA_REPO / "checkpoints" / "mdiqa_shared_5t_4a_mbv3_koniq_xd.pth"
    if p_xd.exists():
        variants.append(VariantSpec("mdiqa_xd", p_xd, local_score=True, semantic_feat="clip_RN50"))

    # Unified
    p_unified = MDIQA_REPO / "checkpoints" / "net_best.pth"
    if p_unified.exists():
        variants.append(VariantSpec("mdiqa_unified", p_unified, semantic_model_name="resnet50"))

    return variants, missing_variants, missing_shared


def inventory_rows(variants: list[VariantSpec], missing_variants: list[str], missing_shared: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in SHARED_CHECKPOINTS.items():
        present, size = file_status(path)
        rows.append(
            {
                "category": "shared",
                "variant": "",
                "role": name,
                "path": str(path),
                "present": present,
                "size_bytes": size,
                "remote_visible_file_id": REMOTE_VISIBLE_FILES.get(path.name, ""),
                "notes": "" if present else "missing shared checkpoint",
            }
        )
    for spec in variants:
        present, size = file_status(spec.fusion_path)
        rows.append(
            {
                "category": "variant_fusion",
                "variant": spec.variant,
                "role": "fusion",
                "path": str(spec.fusion_path),
                "present": present,
                "size_bytes": size,
                "remote_visible_file_id": REMOTE_VISIBLE_FILES.get(spec.fusion_path.name, ""),
                "notes": "selected",
            }
        )
    for name, file_id in sorted(REMOTE_VISIBLE_FILES.items()):
        local_path = MDIQA_REPO / "checkpoints" / name
        if any(row["path"] == str(local_path) for row in rows):
            continue
        present, size = file_status(local_path)
        rows.append(
            {
                "category": "remote_visible_unmapped",
                "variant": "",
                "role": "",
                "path": str(local_path),
                "present": present,
                "size_bytes": size,
                "remote_visible_file_id": file_id,
                "notes": "unmapped",
            }
        )
    return rows


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def failure_row(stage: str, variant: str, dataset: str, image_path: str, error: str) -> dict[str, str]:
    return {
        "stage": stage,
        "variant": variant,
        "dataset": dataset,
        "image_path": image_path,
        "error": error,
    }


def build_sanity_manifest() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_dataset: dict[str, dict[str, dict[str, str]]] = {dataset: {} for dataset in DATASETS}
    failures: list[dict[str, str]] = []
    with MANIFEST_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset = row.get("dataset", "")
            if dataset not in DATASETS:
                continue
            image_path = row.get("image_path", "")
            if not image_path or image_path in by_dataset[dataset]:
                continue
            by_dataset[dataset][image_path] = {
                "dataset": dataset,
                "dataset_label": DATASETS[dataset],
                "split": row.get("split", ""),
                "image_path": image_path,
                "quality_score_0_100": row.get("quality_score_0_100", ""),
            }

    rng = random.Random(SEED)
    sampled: list[dict[str, str]] = []
    for dataset in DATASETS:
        candidates = list(by_dataset[dataset].values())
        rng.shuffle(candidates)
        count = 0
        for row in candidates:
            path = Path(row["image_path"])
            if not path.exists():
                failures.append(failure_row("sampling", "", dataset, row["image_path"], "missing image"))
                continue
            if not is_readable_image(path):
                failures.append(failure_row("sampling", "", dataset, row["image_path"], "unreadable image"))
                continue
            try:
                float(row["quality_score_0_100"])
            except ValueError:
                failures.append(failure_row("sampling", "", dataset, row["image_path"], "invalid target"))
                continue
            sampled.append(row)
            count += 1
            if count == SAMPLE_PER_DATASET:
                break
        if count < SAMPLE_PER_DATASET:
            failures.append(failure_row("sampling", "", dataset, "", f"selected only {count} readable images"))
    sampled.sort(key=lambda row: (row["dataset"], row["image_path"]))
    return sampled, failures


def load_image_tensor(path: Path) -> Any:
    import torch

    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class MDIQAVariantModel:
    def __init__(self, spec: VariantSpec, device: str) -> None:
        import torch

        configure_local_caches()
        install_mdiqa_import_aliases()
        force_timm_no_pretrained()
        from MDIQA.archs.topiq_backbone_head_arch import CFANet_Shared
        from MDIQA.archs.topiq_shared_fusion_arch import MultiDimFusionWeightedMLP

        self.device = torch.device(device)
        technology_paths = [
            str(SHARED_CHECKPOINTS["technology_backbone"]),
            str(SHARED_CHECKPOINTS["technology_head0"]),
            str(SHARED_CHECKPOINTS["technology_head1"]),
            str(SHARED_CHECKPOINTS["technology_head2"]),
            str(SHARED_CHECKPOINTS["technology_head3"]),
            str(SHARED_CHECKPOINTS["technology_head4"]),
        ]
        self.technology_model = CFANet_Shared(
            setting=2,
            num_class=5,
            num_crop=1,
            crop_size=IMAGE_SIZE,
            semantic_model_name="resnet50",
            backbone_pretrain=False,
            pretrained=True,
            pretrained_model_path=technology_paths,
        ).to(self.device)
        self.aesthetic_model = CFANet_Shared(
            setting=2,
            num_class=4,
            num_crop=1,
            crop_size=IMAGE_SIZE,
            semantic_model_name="resnet50",
            backbone_pretrain=False,
            pretrained=False,
            pretrained_model_path=str(SHARED_CHECKPOINTS["aesthetic_color"]),
        ).to(self.device)
        self.fusion_model = MultiDimFusionWeightedMLP(
            num_dim=9,
            aes_num=4,
            dataset_specific_weighted=False,
            instance_specific_weighted=True,
            semantic_model_name=spec.semantic_model_name,
            local_score=spec.local_score,
            semantic_feat=spec.semantic_feat,
        ).to(self.device)
        state = torch.load(spec.fusion_path, map_location="cpu")
        self.fusion_model.load_state_dict(state.get("params", state), strict=True)
        self.technology_model.eval()
        self.aesthetic_model.eval()
        self.fusion_model.eval()

    def predict_batch(self, image_paths: list[Path]) -> list[float]:
        import torch

        batch = torch.stack([load_image_tensor(path) for path in image_paths], dim=0).to(self.device)
        with torch.no_grad():
            tech_global, tech_local, tech_tokens = self.technology_model(batch)
            aes_global, aes_local, aes_tokens = self.aesthetic_model(batch)
            global_scores = torch.cat([*tech_global, *aes_global], dim=1)
            local_scores = [*tech_local, *aes_local]
            final_tokens = [*tech_tokens, *aes_tokens]
            score = self.fusion_model(batch, [global_scores, local_scores, final_tokens])
        return [float(value) for value in score.reshape(-1).detach().cpu().tolist()]


def runtime_info() -> RuntimeInfo:
    import torch

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if cuda_available and device_count else ""
    return RuntimeInfo(
        torch_version=torch.__version__,
        torch_cuda=str(torch.version.cuda),
        cuda_available=cuda_available,
        device_count=device_count,
        device_name=device_name,
        device_used="cuda" if cuda_available else "cpu",
    )


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def linear_aligned_mae(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    if len(pred) < 2 or float(np.std(pred)) == 0.0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(pred, target, 1)
    aligned = slope * pred + intercept
    return float(np.mean(np.abs(aligned - target))), float(slope), float(intercept)


def compute_metrics(predictions: list[dict[str, Any]], failures: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_key[(row["variant"], row["dataset"])].append(row)
    failure_counts = defaultdict(int)
    for row in failures:
        if row["variant"] and row["dataset"]:
            failure_counts[(row["variant"], row["dataset"])] += 1

    metrics: list[dict[str, Any]] = []
    variants = sorted({row["variant"] for row in predictions} | {row["variant"] for row in failures if row["variant"]})
    for variant in variants:
        for dataset, label in DATASETS.items():
            rows = by_key.get((variant, dataset), [])
            raw = np.array([float(row["pred_raw"]) for row in rows], dtype=np.float64)
            pred = np.array([float(row["pred_0_100"]) for row in rows], dtype=np.float64)
            target = np.array([float(row["quality_score_0_100"]) for row in rows], dtype=np.float64)
            aligned_mae, slope, intercept = linear_aligned_mae(pred, target)
            metrics.append(
                {
                    "variant": variant,
                    "dataset": dataset,
                    "dataset_label": label,
                    "n": len(rows),
                    "pred_raw_min": float(np.min(raw)) if len(raw) else float("nan"),
                    "pred_raw_max": float(np.max(raw)) if len(raw) else float("nan"),
                    "pred_raw_mean": float(np.mean(raw)) if len(raw) else float("nan"),
                    "pred_raw_std": float(np.std(raw, ddof=1)) if len(raw) > 1 else float("nan"),
                    "pred_0_100_min": float(np.min(pred)) if len(pred) else float("nan"),
                    "pred_0_100_max": float(np.max(pred)) if len(pred) else float("nan"),
                    "pred_0_100_mean": float(np.mean(pred)) if len(pred) else float("nan"),
                    "pred_0_100_std": float(np.std(pred, ddof=1)) if len(pred) > 1 else float("nan"),
                    "srcc": spearman(pred, target) if len(pred) else float("nan"),
                    "plcc": pearson(pred, target) if len(pred) else float("nan"),
                    "mae_raw_x100": float(np.mean(np.abs(pred - target))) if len(pred) else float("nan"),
                    "mae_linear_aligned": aligned_mae,
                    "linear_slope": slope,
                    "linear_intercept": intercept,
                    "failures": failure_counts[(variant, dataset)],
                }
            )
    return metrics


def best_variant_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, label in DATASETS.items():
        valid = [
            row
            for row in metrics
            if row["dataset"] == dataset and int(row["n"]) > 0 and math.isfinite(float(row["srcc"]))
        ]
        if valid:
            best = max(valid, key=lambda row: float(row["srcc"]))
            srcc = float(best["srcc"])
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": label,
                    "best_variant": best["variant"],
                    "best_srcc": srcc,
                    "production_baseline_srcc": PRODUCTION_BASELINES[dataset],
                    "delta_vs_production": srcc - PRODUCTION_BASELINES[dataset],
                    "n": best["n"],
                    "failures": best["failures"],
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": label,
                    "best_variant": "",
                    "best_srcc": float("nan"),
                    "production_baseline_srcc": PRODUCTION_BASELINES[dataset],
                    "delta_vs_production": float("nan"),
                    "n": 0,
                    "failures": 0,
                }
            )
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value_float):
        return "nan"
    return f"{value_float:.{digits}f}"


def decide(
    metrics: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    missing_variants: list[str],
    missing_shared: list[str],
) -> str:
    if missing_shared or not metrics:
        return "MDIQA_NEEDS_CHECKPOINT_FIX"
    if any(row["stage"] in {"model_load", "inference"} for row in failures):
        return "MDIQA_BENCHMARK_FAILED"
    # Ensure all datasets have at least one valid variant evaluated
    datasets_with_preds = {row["dataset"] for row in metrics if int(row["n"]) > 0}
    if len(datasets_with_preds) < len(DATASETS):
        return "MDIQA_BENCHMARK_FAILED"
    wins = [
        float(row["best_srcc"]) > float(row["production_baseline_srcc"])
        for row in best_rows
        if math.isfinite(float(row["best_srcc"]))
    ]
    if len(wins) == len(DATASETS) and all(wins):
        return "MDIQA_READY_FOR_FULL_BENCHMARK"
    if any(wins):
        return "MDIQA_PARTIAL_TEACHER_CANDIDATE"
    return "MDIQA_NOT_PROMISING"


def manual_download_text(missing_variants: list[str]) -> list[str]:
    if not missing_variants:
        return ["- All mapped variants are present."]
    lines = [
        "- Official folder: `https://drive.google.com/drive/folders/1-RajAoPc-0v-WWQ9BnYTi-HxbpJNLZcl?usp=sharing`.",
        "- Specific SPAQ/FLIVE fusion models (e.g., `MDIQA_SPAQ.pth`) were NOT found in the official Drive listing.",
        "- The Cross-Dataset model (`mdiqa_shared_5t_4a_mbv3_koniq_xd.pth`) and Unified Best model (`net_best.pth`) are intended as the official representatives for these datasets.",
        "- Ensure these files are present in `MDIQA/checkpoints/`:",
    ]
    for variant, candidates in VARIANT_FUSION_CANDIDATES.items():
        for path in candidates:
            lines.append(f"  - `{path.name}`")
    return lines


def write_report(
    manifest_rows: list[dict[str, str]],
    inventory: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    rt: RuntimeInfo,
    decision: str,
    elapsed: float,
    variants: list[VariantSpec],
    missing_variants: list[str],
    missing_shared: list[str],
) -> None:
    counts = defaultdict(int)
    split_counts = defaultdict(int)
    for row in manifest_rows:
        counts[row["dataset"]] += 1
        split_counts[(row["dataset"], row["split"])] += 1

    lines = [
        "# MDIQA Variant 100-image Sanity Benchmark Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        f"- Evaluated variants: `{', '.join(spec.variant for spec in variants) if variants else 'none'}`.",
        f"- Evaluated {len(manifest_rows)} sampled images with {len(predictions)} predictions.",
        f"- Elapsed seconds: `{elapsed:.1f}`.",
        "- Scope stayed benchmark-only: no training, TFLite export, Flutter edits, production scoring edits, distillation, teacher-label creation, or full benchmark.",
        "",
        "## 2. Inputs",
        f"- Input manifest: `{MANIFEST_CSV}`.",
        f"- Output directory: `{OUTPUT_DIR}`.",
        f"- Seed: `{SEED}`.",
        f"- Sample target: `{SAMPLE_PER_DATASET}` per dataset.",
        f"- Batch size: `{BATCH_SIZE}`.",
        "",
        "## 3. Checkpoint Inventory",
        f"- Inventory CSV: `{INVENTORY_CSV}`.",
        f"- Missing required variants: `{', '.join(missing_variants) if missing_variants else 'none'}`.",
        f"- Missing shared checkpoints: `{len(missing_shared)}`.",
    ]
    for row in inventory:
        if row["category"] in {"missing_required_variant", "missing_required_shared"}:
            lines.append(f"- {row['category']}: {row['variant']} {row['path']} {row['notes']}")

    lines.extend(
        [
            "",
            "## 4. CUDA / Environment Notes",
            f"- CUDA report: `{CUDA_REPORT}`.",
            f"- torch: `{rt.torch_version}`.",
            f"- torch.version.cuda: `{rt.torch_cuda}`.",
            f"- torch.cuda.is_available: `{rt.cuda_available}`.",
            f"- torch.cuda.device_count: `{rt.device_count}`.",
            f"- device used: `{rt.device_used}`.",
            f"- device name: `{rt.device_name}`.",
            "- Use clean command pattern: `env -u LD_LIBRARY_PATH ./.venv_mdiqa/bin/python scripts/technical_iqa/run_mdiqa_variant_sanity_benchmark.py`.",
            "",
            "## 5. Dataset Sampling",
        ]
    )
    for dataset, label in DATASETS.items():
        lines.append(f"- {label}: {counts[dataset]} images.")
        for (split_dataset, split), count in sorted(split_counts.items()):
            if split_dataset == dataset:
                lines.append(f"  - {split}: {count}")

    lines.extend(["", "## 6. Variant Prediction Distributions"])
    if metrics:
        for row in metrics:
            lines.append(
                f"- {row['variant']} / {row['dataset_label']}: raw min={fmt(row['pred_raw_min'])}, "
                f"max={fmt(row['pred_raw_max'])}, mean={fmt(row['pred_raw_mean'])}, "
                f"std={fmt(row['pred_raw_std'])}; 0..100 mean={fmt(row['pred_0_100_mean'])}"
            )
    else:
        lines.append("- No predictions; required checkpoint variants are missing.")

    lines.extend(["", "## 7. Dataset-wise Metrics"])
    lines.append("| Variant | Dataset | n | SRCC | PLCC | MAE raw*100 | Linear MAE | Failures |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in metrics:
        lines.append(
            f"| {row['variant']} | {row['dataset_label']} | {row['n']} | {fmt(row['srcc'])} | "
            f"{fmt(row['plcc'])} | {fmt(row['mae_raw_x100'])} | "
            f"{fmt(row['mae_linear_aligned'])} | {row['failures']} |"
        )

    lines.extend(["", "## 8. Best Variant by Dataset"])
    lines.append("| Dataset | Best Variant | Best SRCC | Production SRCC | Delta |")
    lines.append("|---|---|---:|---:|---:|")
    for row in best_rows:
        lines.append(
            f"| {row['dataset_label']} | {row['best_variant']} | {fmt(row['best_srcc'])} | "
            f"{fmt(row['production_baseline_srcc'], 3)} | {fmt(row['delta_vs_production'])} |"
        )

    lines.extend(["", "## 9. Comparison Against Production Baseline"])
    if best_rows and any(row["best_variant"] for row in best_rows):
        for row in best_rows:
            lines.append(
                f"- {row['dataset_label']}: best `{row['best_variant']}` SRCC {fmt(row['best_srcc'])} "
                f"vs production {fmt(row['production_baseline_srcc'], 3)} "
                f"(delta {fmt(row['delta_vs_production'])})."
            )
    else:
        lines.append("- Not evaluated because expected SPAQ/FLIVE checkpoints are missing.")

    lines.extend(["", "## 10. Failure Cases"])
    if failures:
        for row in failures[:50]:
            lines.append(f"- {row['stage']} | {row['variant']} | {row['dataset']} | {row['image_path']} | {row['error']}")
        if len(failures) > 50:
            lines.append(f"- ... {len(failures) - 50} more failures in `failures.csv`.")
    else:
        lines.append("- None.")

    lines.extend(["", "## 11. Final Decision", decision, "", "## 12. Recommended Next Step"])
    if decision == "MDIQA_NEEDS_CHECKPOINT_FIX":
        lines.extend(manual_download_text(missing_variants))
    elif decision == "MDIQA_READY_FOR_FULL_BENCHMARK":
        lines.append("- Run the full benchmark with the best dataset-specific variant mapping and clean CUDA environment.")
    elif decision == "MDIQA_PARTIAL_TEACHER_CANDIDATE":
        lines.append("- Keep only datasets whose best variant beats production for the next bounded check; do not create labels yet.")
    elif decision == "MDIQA_NOT_PROMISING":
        lines.append("- Do not start distillation or teacher-label creation from these variants.")
    else:
        lines.append("- Fix the benchmark blocker and rerun only this sanity benchmark.")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_local_caches()
    variants, missing_variants, missing_shared = resolve_variants()
    inventory = inventory_rows(variants, missing_variants, missing_shared)
    manifest_rows, failures = build_sanity_manifest()
    write_csv(VARIANT_MANIFEST_CSV, manifest_rows, ["dataset", "dataset_label", "split", "image_path", "quality_score_0_100"])
    write_csv(
        INVENTORY_CSV,
        inventory,
        ["category", "variant", "role", "path", "present", "size_bytes", "remote_visible_file_id", "notes"],
    )

    rt = runtime_info()
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    if missing_shared or not variants:
        for variant in missing_variants:
            failures.append(failure_row("checkpoint_inventory", variant, "", "", "missing fusion checkpoint"))
        for item in missing_shared:
            failures.append(failure_row("checkpoint_inventory", "", "", item, "missing required shared checkpoint"))
        decision = "MDIQA_NEEDS_CHECKPOINT_FIX"
    else:
        for spec in variants:
            try:
                model = MDIQAVariantModel(spec, rt.device_used)
            except Exception:
                failures.append(failure_row("model_load", spec.variant, "", str(spec.fusion_path), traceback.format_exc(limit=20)))
                continue
            for start_idx in range(0, len(manifest_rows), BATCH_SIZE):
                batch_rows = manifest_rows[start_idx : start_idx + BATCH_SIZE]
                batch_paths = [Path(row["image_path"]) for row in batch_rows]
                try:
                    scores = model.predict_batch(batch_paths)
                    for row, score in zip(batch_rows, scores):
                        predictions.append(
                            {
                                "variant": spec.variant,
                                "dataset": row["dataset"],
                                "dataset_label": row["dataset_label"],
                                "split": row["split"],
                                "image_path": row["image_path"],
                                "quality_score_0_100": row["quality_score_0_100"],
                                "pred_raw": score,
                                "pred_0_100": score * 100.0,
                            }
                        )
                except Exception:
                    error = traceback.format_exc(limit=12)
                    for row in batch_rows:
                        failures.append(failure_row("inference", spec.variant, row["dataset"], row["image_path"], error))
                print(f"{spec.variant}: scored {min(start_idx + BATCH_SIZE, len(manifest_rows))}/{len(manifest_rows)}")
        metrics = compute_metrics(predictions, failures)
        best_rows = best_variant_rows(metrics)
        decision = decide(metrics, best_rows, failures, missing_variants, missing_shared)

    if not metrics:
        best_rows = best_variant_rows(metrics)
    write_csv(
        PREDICTIONS_CSV,
        predictions,
        ["variant", "dataset", "dataset_label", "split", "image_path", "quality_score_0_100", "pred_raw", "pred_0_100"],
    )
    write_csv(
        METRICS_CSV,
        metrics,
        [
            "variant",
            "dataset",
            "dataset_label",
            "n",
            "pred_raw_min",
            "pred_raw_max",
            "pred_raw_mean",
            "pred_raw_std",
            "pred_0_100_min",
            "pred_0_100_max",
            "pred_0_100_mean",
            "pred_0_100_std",
            "srcc",
            "plcc",
            "mae_raw_x100",
            "mae_linear_aligned",
            "linear_slope",
            "linear_intercept",
            "failures",
        ],
    )
    write_csv(
        BEST_VARIANT_CSV,
        best_rows,
        ["dataset", "dataset_label", "best_variant", "best_srcc", "production_baseline_srcc", "delta_vs_production", "n", "failures"],
    )
    write_csv(FAILURES_CSV, failures, ["stage", "variant", "dataset", "image_path", "error"])
    elapsed = time.perf_counter() - start
    write_report(
        manifest_rows,
        inventory,
        predictions,
        metrics,
        best_rows,
        failures,
        rt,
        decision,
        elapsed,
        variants,
        missing_variants,
        missing_shared,
    )
    print(f"Wrote {VARIANT_MANIFEST_CSV}")
    print(f"Wrote {INVENTORY_CSV}")
    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {METRICS_CSV}")
    print(f"Wrote {BEST_VARIANT_CSV}")
    print(f"Wrote {FAILURES_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Final decision: {decision}")


if __name__ == "__main__":
    main()
