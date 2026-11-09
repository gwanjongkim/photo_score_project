# MDIQA Koniq checkpoint로 100장 단위 sanity benchmark를 실행한다.
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
OPEN_CLIP_REPO = ROOT / "open_clip"
MANIFEST_CSV = (
    ROOT
    / "outputs"
    / "research_protocol_iqa_manifests_20260530"
    / "benchmark_all_research_protocol_iqa.csv"
)
CUDA_REPORT = ROOT / "outputs" / "mdiqa_cuda_diagnosis_20260601" / "report.md"
OUTPUT_DIR = ROOT / "outputs" / "mdiqa_sanity_20260601"
SANITY_MANIFEST_CSV = OUTPUT_DIR / "sanity_manifest.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
METRICS_CSV = OUTPUT_DIR / "dataset_metrics.csv"
FAILURES_CSV = OUTPUT_DIR / "failures.csv"
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
CHECKPOINTS = {
    "fusion": MDIQA_REPO / "checkpoints" / "mdiqa_shared_5t_4a_mbv3_koniq_seed1.pth",
    "technology_backbone": MDIQA_REPO / "checkpoints" / "net_freeze_backbone.pth",
    "technology_head0": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head0_head0.pth",
    "technology_head1": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head1_head1.pth",
    "technology_head2": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head2_head2.pth",
    "technology_head3": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head3_head3.pth",
    "technology_head4": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head4_head4.pth",
    "aesthetic_color": MDIQA_REPO / "checkpoints" / "net_best_PARA_val_color.pth",
}


@dataclass
class RuntimeInfo:
    torch_version: str
    torch_cuda: str
    cuda_available: bool
    device_count: int
    device_name: str
    device_used: str


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


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def failure_row(stage: str, dataset: str, image_path: str, error: str) -> dict[str, str]:
    return {
        "stage": stage,
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
                failures.append(failure_row("sampling", dataset, row["image_path"], "missing image"))
                continue
            if not is_readable_image(path):
                failures.append(failure_row("sampling", dataset, row["image_path"], "unreadable image"))
                continue
            try:
                float(row["quality_score_0_100"])
            except ValueError:
                failures.append(failure_row("sampling", dataset, row["image_path"], "invalid target"))
                continue
            sampled.append(row)
            count += 1
            if count == SAMPLE_PER_DATASET:
                break
        if count < SAMPLE_PER_DATASET:
            failures.append(
                failure_row("sampling", dataset, "", f"selected only {count} readable images")
            )
    sampled.sort(key=lambda row: (row["dataset"], row["image_path"]))
    return sampled, failures


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def missing_checkpoints() -> list[Path]:
    return [path for path in CHECKPOINTS.values() if not path.exists() or path.stat().st_size < 1024]


def load_image_tensor(path: Path) -> Any:
    import torch

    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class MDIQAKoniqModel:
    def __init__(self, device: str) -> None:
        import torch

        configure_local_caches()
        install_mdiqa_import_aliases()
        force_timm_no_pretrained()

        from MDIQA.archs.topiq_backbone_head_arch import CFANet_Shared
        from MDIQA.archs.topiq_shared_fusion_arch import MultiDimFusionWeightedMLP

        self.device = torch.device(device)
        technology_paths = [
            str(CHECKPOINTS["technology_backbone"]),
            str(CHECKPOINTS["technology_head0"]),
            str(CHECKPOINTS["technology_head1"]),
            str(CHECKPOINTS["technology_head2"]),
            str(CHECKPOINTS["technology_head3"]),
            str(CHECKPOINTS["technology_head4"]),
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
            pretrained_model_path=str(CHECKPOINTS["aesthetic_color"]),
        ).to(self.device)
        self.fusion_model = MultiDimFusionWeightedMLP(
            num_dim=9,
            aes_num=4,
            dataset_specific_weighted=False,
            instance_specific_weighted=True,
            semantic_model_name="mobilenetv3_large_100.ra_in1k",
            local_score=False,
            semantic_feat="without",
        ).to(self.device)
        state = torch.load(CHECKPOINTS["fusion"], map_location="cpu")
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
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_dataset[row["dataset"]].append(row)
    failure_counts = defaultdict(int)
    for row in failures:
        if row["dataset"]:
            failure_counts[row["dataset"]] += 1

    metrics: list[dict[str, Any]] = []
    for dataset, label in DATASETS.items():
        rows = by_dataset.get(dataset, [])
        raw = np.array([float(row["pred_raw"]) for row in rows], dtype=np.float64)
        pred = np.array([float(row["pred_0_100"]) for row in rows], dtype=np.float64)
        target = np.array([float(row["quality_score_0_100"]) for row in rows], dtype=np.float64)
        aligned_mae, slope, intercept = linear_aligned_mae(pred, target)
        metrics.append(
            {
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
                "failures": failure_counts[dataset],
            }
        )
    return metrics


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value_float):
        return "nan"
    return f"{value_float:.{digits}f}"


def decide(metrics: list[dict[str, Any]], failures: list[dict[str, str]], rt: RuntimeInfo, elapsed: float) -> str:
    if any(row["stage"] in {"model_load", "inference"} for row in failures):
        return "MDIQA_BENCHMARK_FAILED"
    valid = [row for row in metrics if int(row["n"]) >= 80 and math.isfinite(float(row["srcc"]))]
    if len(valid) < len(DATASETS):
        return "MDIQA_BENCHMARK_FAILED"
    if not rt.cuda_available and elapsed > 1800:
        return "MDIQA_NEEDS_CUDA_FIX_FIRST"

    by_dataset = {row["dataset"]: float(row["srcc"]) for row in metrics}
    koniq_ok = by_dataset.get("koniq10k", -1.0) >= 0.70 * PRODUCTION_BASELINES["koniq10k"]
    flive_ok = by_dataset.get("flive", -1.0) >= 0.70 * PRODUCTION_BASELINES["flive"]
    spaq_ok = by_dataset.get("spaq", -1.0) >= 0.70 * PRODUCTION_BASELINES["spaq"]
    if koniq_ok and not (flive_ok and spaq_ok):
        return "MDIQA_NEEDS_VARIANT_CHECKPOINTS"
    if any(score >= 0.70 * PRODUCTION_BASELINES[dataset] for dataset, score in by_dataset.items()):
        return "MDIQA_READY_FOR_FULL_BENCHMARK"
    return "MDIQA_NOT_PROMISING"


def write_report(
    manifest_rows: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    failures: list[dict[str, str]],
    rt: RuntimeInfo,
    decision: str,
    elapsed: float,
) -> None:
    counts = defaultdict(int)
    split_counts = defaultdict(int)
    for row in manifest_rows:
        counts[row["dataset"]] += 1
        split_counts[(row["dataset"], row["split"])] += 1

    lines = [
        "# MDIQA 100-image Sanity Benchmark Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        f"- Evaluated {len(manifest_rows)} sampled images with {len(predictions)} predictions.",
        f"- Elapsed seconds: `{elapsed:.1f}`.",
        "- Scope stayed benchmark-only: no training, TFLite export, Flutter edits, production scoring edits, distillation, or teacher-label creation.",
        "",
        "## 2. Inputs",
        f"- Input manifest: `{MANIFEST_CSV}`.",
        f"- Output directory: `{OUTPUT_DIR}`.",
        f"- Seed: `{SEED}`.",
        f"- Sample target: `{SAMPLE_PER_DATASET}` per dataset.",
        f"- Batch size: `{BATCH_SIZE}`.",
        "",
        "## 3. CUDA Diagnosis Summary",
        f"- CUDA report: `{CUDA_REPORT}`.",
        f"- torch: `{rt.torch_version}`.",
        f"- torch.version.cuda: `{rt.torch_cuda}`.",
        f"- torch.cuda.is_available: `{rt.cuda_available}`.",
        f"- torch.cuda.device_count: `{rt.device_count}`.",
        f"- device used: `{rt.device_used}`.",
        f"- device name: `{rt.device_name}`.",
        "- Torch reinstall needed: `no`.",
        "",
        "## 4. Checkpoint Used",
        "- Variant: `Koniq MDIQA checkpoint set`.",
        *[f"- {name}: `{path}` ({path.stat().st_size if path.exists() else 0} bytes)" for name, path in CHECKPOINTS.items()],
        "",
        "## 5. Dataset Sampling",
    ]
    for dataset, label in DATASETS.items():
        lines.append(f"- {label}: {counts[dataset]} images.")
        for (split_dataset, split), count in sorted(split_counts.items()):
            if split_dataset == dataset:
                lines.append(f"  - {split}: {count}")

    lines.extend(["", "## 6. Raw Prediction Distributions"])
    for row in metrics:
        lines.append(
            f"- {row['dataset_label']}: raw min={fmt(row['pred_raw_min'])}, "
            f"max={fmt(row['pred_raw_max'])}, mean={fmt(row['pred_raw_mean'])}, "
            f"std={fmt(row['pred_raw_std'])}; 0..100 mean={fmt(row['pred_0_100_mean'])}"
        )

    lines.extend(["", "## 7. Dataset-wise Metrics"])
    lines.append("| Dataset | n | SRCC | PLCC | MAE raw*100 | Linear MAE | Failures |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in metrics:
        lines.append(
            f"| {row['dataset_label']} | {row['n']} | {fmt(row['srcc'])} | "
            f"{fmt(row['plcc'])} | {fmt(row['mae_raw_x100'])} | "
            f"{fmt(row['mae_linear_aligned'])} | {row['failures']} |"
        )

    lines.extend(["", "## 8. Comparison Against Production Baseline"])
    lines.append("| Dataset | Production SRCC | MDIQA Koniq SRCC | Delta |")
    lines.append("|---|---:|---:|---:|")
    for row in metrics:
        baseline = PRODUCTION_BASELINES[row["dataset"]]
        srcc = float(row["srcc"])
        lines.append(f"| {row['dataset_label']} | {baseline:.3f} | {fmt(srcc)} | {fmt(srcc - baseline)} |")

    lines.extend(["", "## 9. Failure Cases"])
    if failures:
        for row in failures[:50]:
            lines.append(f"- {row['stage']} | {row['dataset']} | {row['image_path']} | {row['error']}")
        if len(failures) > 50:
            lines.append(f"- ... {len(failures) - 50} more failures in `failures.csv`.")
    else:
        lines.append("- None.")

    lines.extend(["", "## 10. Final Decision", decision, "", "## 11. Recommended Next Step"])
    if decision == "MDIQA_READY_FOR_FULL_BENCHMARK":
        lines.append("- Run the full benchmark using the current Koniq checkpoint set; keep CUDA environment clean.")
    elif decision == "MDIQA_NEEDS_CUDA_FIX_FIRST":
        lines.append("- Fix CUDA runtime visibility before running any larger MDIQA job.")
    elif decision == "MDIQA_NEEDS_VARIANT_CHECKPOINTS":
        lines.append("- Download/check SPAQ and FLIVE MDIQA checkpoints before judging MDIQA as a multi-dataset teacher.")
    elif decision == "MDIQA_NOT_PROMISING":
        lines.append("- Do not start distillation or teacher-label creation from this checkpoint.")
    else:
        lines.append("- Fix the benchmark blocker and rerun only this sanity benchmark.")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_local_caches()
    manifest_rows, failures = build_sanity_manifest()
    write_csv(SANITY_MANIFEST_CSV, manifest_rows, ["dataset", "dataset_label", "split", "image_path", "quality_score_0_100"])

    missing = missing_checkpoints()
    predictions: list[dict[str, Any]] = []
    if missing:
        for path in missing:
            failures.append(failure_row("model_load", "", str(path), "missing checkpoint"))
        rt = runtime_info()
        metrics = compute_metrics(predictions, failures)
        decision = "MDIQA_BENCHMARK_FAILED"
    else:
        rt = runtime_info()
        try:
            model = MDIQAKoniqModel(rt.device_used)
            for start_idx in range(0, len(manifest_rows), BATCH_SIZE):
                batch_rows = manifest_rows[start_idx : start_idx + BATCH_SIZE]
                batch_paths = [Path(row["image_path"]) for row in batch_rows]
                try:
                    scores = model.predict_batch(batch_paths)
                    for row, score in zip(batch_rows, scores):
                        predictions.append(
                            {
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
                        failures.append(failure_row("inference", row["dataset"], row["image_path"], error))
                print(f"scored {min(start_idx + BATCH_SIZE, len(manifest_rows))}/{len(manifest_rows)}")
        except Exception:
            failures.append(failure_row("model_load", "", "", traceback.format_exc(limit=20)))
        metrics = compute_metrics(predictions, failures)
        elapsed_for_decision = time.perf_counter() - start
        decision = decide(metrics, failures, rt, elapsed_for_decision)

    write_csv(
        PREDICTIONS_CSV,
        predictions,
        ["dataset", "dataset_label", "split", "image_path", "quality_score_0_100", "pred_raw", "pred_0_100"],
    )
    write_csv(
        METRICS_CSV,
        metrics,
        [
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
    write_csv(FAILURES_CSV, failures, ["stage", "dataset", "image_path", "error"])
    elapsed = time.perf_counter() - start
    write_report(manifest_rows, predictions, metrics, failures, rt, decision, elapsed)
    print(f"Wrote {SANITY_MANIFEST_CSV}")
    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {METRICS_CSV}")
    print(f"Wrote {FAILURES_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Final decision: {decision}")


if __name__ == "__main__":
    main()
