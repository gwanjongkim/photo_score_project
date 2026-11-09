# MDIQA FLIVE image-only test split sanity benchmark.
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

# Force GPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ROOT = Path(__file__).resolve().parents[2]
MDIQA_REPO = ROOT / "MDIQA"
MANIFEST_CSV = (
    ROOT
    / "outputs"
    / "research_protocol_iqa_manifests_20260530"
    / "benchmark_all_research_protocol_iqa.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "mdiqa_flive_image_test_sanity_20260601"
FLIVE_MANIFEST_CSV = OUTPUT_DIR / "flive_image_test_manifest.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
METRICS_CSV = OUTPUT_DIR / "metrics.csv"
FAILURES_CSV = OUTPUT_DIR / "failures.csv"
SENSITIVITY_CSV = OUTPUT_DIR / "sensitivity_checks.csv"
REPORT_MD = OUTPUT_DIR / "report.md"

SEED = 20260601
SAMPLE_LIMIT = 1000
SUBSET_LIMIT = 100
BATCH_SIZE = 8
IMAGE_SIZE = 384

PRODUCTION_FLIVE_SRCC = 0.574
PAPER_FLIVE_SRCC = 0.612

SHARED_CHECKPOINTS = {
    "technology_backbone": MDIQA_REPO / "checkpoints" / "net_freeze_backbone.pth",
    "technology_head0": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head0_head0.pth",
    "technology_head1": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head1_head1.pth",
    "technology_head2": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head2_head2.pth",
    "technology_head3": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head3_head3.pth",
    "technology_head4": MDIQA_REPO / "checkpoints" / "net_best_koniq10k_val_head4_head4.pth",
    "aesthetic_color": MDIQA_REPO / "checkpoints" / "net_best_PARA_val_color.pth",
}

VARIANT_XD = {
    "variant": "mdiqa_xd",
    "fusion_path": MDIQA_REPO / "checkpoints" / "mdiqa_shared_5t_4a_mbv3_koniq_xd.pth",
    "local_score": True,
    "semantic_model_name": "mobilenetv3_large_100.ra_in1k",
    "semantic_feat": "clip_RN50",
}


@dataclass
class CUDAInfo:
    torch_version: str
    cuda_version: str
    available: bool
    device_count: int
    device_name: str
    cudnn_version: int | str


def cuda_preflight() -> CUDAInfo:
    import torch

    return CUDAInfo(
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count(),
        device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "N/A",
        cudnn_version=torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A",
    )


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


def load_image_tensor(path: Path) -> Any:
    import torch

    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class MDIQAVariantModel:
    def __init__(self, spec: dict[str, Any], device: str) -> None:
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
            semantic_model_name=spec["semantic_model_name"],
            local_score=spec["local_score"],
            semantic_feat=spec["semantic_feat"],
        ).to(self.device)
        state = torch.load(spec["fusion_path"], map_location="cpu")
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


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def build_filtered_manifest() -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    filter_rule = "FLIVE only, content_type == image, split == test"

    with MANIFEST_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") != "flive":
                continue
            
            # Content type filter
            ct = row.get("content_type", "")
            if not ct:
                # Infer from path
                if "/patches/" in row["image_path"] or "_patch_" in row["image_path"]:
                    continue
            elif ct != "image":
                continue
            
            # Split filter
            if row.get("split") != "test":
                continue

            image_path = row.get("image_path", "")
            if not image_path:
                continue

            rows.append(row)

    rng = random.Random(SEED)
    rng.shuffle(rows)
    
    sampled: list[dict[str, str]] = []
    for row in rows:
        path = Path(row["image_path"])
        if not path.exists():
            failures.append({"stage": "sampling", "image_path": row["image_path"], "error": "missing image"})
            continue
        if not is_readable_image(path):
            failures.append({"stage": "sampling", "image_path": row["image_path"], "error": "unreadable image"})
            continue
        try:
            float(row["quality_score_0_100"])
        except ValueError:
            failures.append({"stage": "sampling", "image_path": row["image_path"], "error": "invalid target"})
            continue
        
        sampled.append(row)
        if len(sampled) >= SAMPLE_LIMIT:
            break

    return sampled, failures, filter_rule


def compute_aggregate_metrics(predictions: list[dict[str, Any]], limit: int | None = None) -> dict[str, Any]:
    rows = predictions[:limit] if limit else predictions
    if not rows:
        return {}
    
    raw = np.array([float(row["pred_raw"]) for row in rows])
    pred_0_100 = np.array([float(row["pred_0_100"]) for row in rows])
    mos = np.array([float(row["quality_score_0_100"]) for row in rows])
    
    srcc = spearman(pred_0_100, mos)
    plcc = pearson(pred_0_100, mos)
    mae_raw_x100 = np.mean(np.abs(pred_0_100 - mos))
    aligned_mae, _, _ = linear_aligned_mae(pred_0_100, mos)
    
    return {
        "n": len(rows),
        "srcc": srcc,
        "plcc": plcc,
        "mae_raw_x100": mae_raw_x100,
        "mae_linear_aligned": aligned_mae,
        "pred_raw_min": np.min(raw),
        "pred_raw_max": np.max(raw),
        "pred_raw_mean": np.mean(raw),
        "pred_raw_std": np.std(raw, ddof=1) if len(raw) > 1 else 0,
        "pred_0_100_min": np.min(pred_0_100),
        "pred_0_100_max": np.max(pred_0_100),
        "pred_0_100_mean": np.mean(pred_0_100),
        "pred_0_100_std": np.std(pred_0_100, ddof=1) if len(pred_0_100) > 1 else 0,
        "mos_min": np.min(mos),
        "mos_max": np.max(mos),
        "mos_mean": np.mean(mos),
        "mos_std": np.std(mos, ddof=1) if len(mos) > 1 else 0,
    }


def write_report(
    cuda: CUDAInfo,
    filter_rule: str,
    manifest_n: int,
    predictions: list[dict[str, Any]],
    failures: list[dict[str, str]],
    elapsed: float,
    decision: str,
) -> None:
    metrics_all = compute_aggregate_metrics(predictions)
    metrics_100 = compute_aggregate_metrics(predictions, SUBSET_LIMIT) if len(predictions) >= SUBSET_LIMIT else {}

    lines = [
        "# MDIQA FLIVE Image-Test Sanity Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        f"- N evaluated: `{len(predictions)}`.",
        f"- Elapsed seconds: `{elapsed:.1f}`.",
        "",
        "## 2. Inputs",
        f"- Input manifest: `{MANIFEST_CSV}`.",
        f"- Output directory: `{OUTPUT_DIR}`.",
        "",
        "## 3. CUDA Preflight",
        f"- torch version: `{cuda.torch_version}`.",
        f"- CUDA version: `{cuda.cuda_version}`.",
        f"- CUDA available: `{cuda.available}`.",
        f"- Device count: `{cuda.device_count}`.",
        f"- Device name: `{cuda.device_name}`.",
        f"- cuDNN version: `{cuda.cudnn_version}`.",
        "",
        "## 4. FLIVE Filtering Rule",
        f"- Rule: `{filter_rule}`.",
        "",
        "## 5. Dataset Counts",
        f"- Valid candidates found: `{manifest_n}`.",
        f"- Sample target: `{SAMPLE_LIMIT}`.",
        f"- Successfully evaluated: `{len(predictions)}`.",
        "",
        "## 6. MDIQA Variant Used",
        f"- Variant: `{VARIANT_XD['variant']}`.",
        f"- Checkpoint: `{VARIANT_XD['fusion_path'].name}`.",
        "",
        "## 7. Prediction Distribution",
    ]
    
    if metrics_all:
        lines.extend([
            f"- Raw min/max/mean: `{metrics_all['pred_raw_min']:.4f} / {metrics_all['pred_raw_max']:.4f} / {metrics_all['pred_raw_mean']:.4f}`.",
            f"- 0..100 min/max/mean: `{metrics_all['pred_0_100_min']:.2f} / {metrics_all['pred_0_100_max']:.2f} / {metrics_all['pred_0_100_mean']:.2f}`.",
            f"- MOS min/max/mean: `{metrics_all['mos_min']:.2f} / {metrics_all['mos_max']:.2f} / {metrics_all['mos_mean']:.2f}`.",
        ])
    else:
        lines.append("- No metrics available.")

    lines.extend(["", "## 8. Metrics Against MOS"])
    if metrics_all:
        lines.extend([
            f"- **All (n={metrics_all['n']}):**",
            f"  - SRCC: `{metrics_all['srcc']:.4f}`",
            f"  - PLCC: `{metrics_all['plcc']:.4f}`",
            f"  - MAE (0..100): `{metrics_all['mae_raw_x100']:.4f}`",
            f"  - Linear Aligned MAE: `{metrics_all['mae_linear_aligned']:.4f}`",
        ])
    
    if metrics_100:
        lines.extend([
            f"- **Subset (n={metrics_100['n']}):**",
            f"  - SRCC: `{metrics_100['srcc']:.4f}`",
            f"  - PLCC: `{metrics_100['plcc']:.4f}`",
        ])

    lines.extend(["", "## 9. Comparison Against Production and Paper"])
    if metrics_all:
        d_prod = metrics_all["srcc"] - PRODUCTION_FLIVE_SRCC
        d_paper = metrics_all["srcc"] - PAPER_FLIVE_SRCC
        lines.extend([
            f"- Production FLIVE SRCC: `{PRODUCTION_FLIVE_SRCC}`",
            f"- Paper FLIVE SRCC: `{PAPER_FLIVE_SRCC}`",
            f"- Delta vs Production: `{d_prod:+.4f}`",
            f"- Delta vs Paper: `{d_paper:+.4f}`",
        ])

    lines.extend(["", "## 10. Sensitivity Checks"])
    if predictions:
        pred = np.array([p["pred_raw"] for p in predictions])
        mos = np.array([p["quality_score_0_100"] for p in predictions])
        lines.extend([
            f"- SRCC(pred, MOS): `{spearman(pred, mos):.4f}`",
            f"- SRCC(-pred, MOS): `{spearman(-pred, mos):.4f}`",
            f"- SRCC(pred, 100-MOS): `{spearman(pred, 100-mos):.4f}`",
        ])
        
        # Source group aggregation if image_id suggests groups
        group_scores = defaultdict(list)
        group_mos = {}
        for p in predictions:
            # Infer source image from path/id
            # Path usually: /.../blur_dataset/motion0044.jpg
            # For FLIVE image-only, each image is usually unique.
            img_id = Path(p["image_path"]).name
            group_scores[img_id].append(p["pred_raw"])
            group_mos[img_id] = p["quality_score_0_100"]
        
        if len(group_scores) < len(predictions):
            agg_pred = []
            agg_mos = []
            for gid in group_scores:
                agg_pred.append(np.mean(group_scores[gid]))
                agg_mos.append(group_mos[gid])
            lines.append(f"- SRCC(aggregated by filename, n={len(agg_pred)}): `{spearman(np.array(agg_pred), np.array(agg_mos)):.4f}`")
        else:
            lines.append("- Source group aggregation: N/A (all images unique).")

    lines.extend(["", "## 11. Failure Cases"])
    if failures:
        lines.append(f"- Total failures: `{len(failures)}`.")
        for f in failures[:10]:
            lines.append(f"  - {f['stage']} | {f['image_path']} | {f['error']}")
    else:
        lines.append("- None.")

    lines.extend(["", "## 12. Final Decision", decision, "", "## 13. Recommended Next Step"])
    if decision == "MDIQA_FLIVE_READY_FOR_FULL_IMAGE_BENCHMARK":
        lines.append("- Proceed with full FLIVE image test set benchmark.")
    elif decision == "MDIQA_FLIVE_PROMISING_BUT_NEEDS_FULL_TEST":
        lines.append("- Check if larger sample improves SRCC stability.")
    elif decision == "MDIQA_FLIVE_BENCHMARK_FAILED_CPU_FALLBACK_BLOCKED":
        lines.append("- Ensure environment is clean and use correct python binary with env vars.")
    else:
        lines.append("- Investigate remaining performance gap vs production.")
    
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    cuda = cuda_preflight()
    if not cuda.available:
        decision = "MDIQA_FLIVE_BENCHMARK_FAILED_CPU_FALLBACK_BLOCKED"
        write_report(cuda, "N/A", 0, [], [], 0.0, decision)
        print(f"CUDA unavailable. Decision: {decision}")
        return

    sampled_rows, sampling_failures, filter_rule = build_filtered_manifest()
    
    with FLIVE_MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        if sampled_rows:
            writer = csv.DictWriter(f, fieldnames=sampled_rows[0].keys())
            writer.writeheader()
            writer.writerows(sampled_rows)

    predictions = []
    failures = sampling_failures
    
    try:
        model = MDIQAVariantModel(VARIANT_XD, "cuda")
        for start_idx in range(0, len(sampled_rows), BATCH_SIZE):
            batch_rows = sampled_rows[start_idx : start_idx + BATCH_SIZE]
            batch_paths = [Path(row["image_path"]) for row in batch_rows]
            try:
                scores = model.predict_batch(batch_paths)
                for row, score in zip(batch_rows, scores):
                    predictions.append({
                        "image_path": row["image_path"],
                        "quality_score_0_100": float(row["quality_score_0_100"]),
                        "pred_raw": score,
                        "pred_0_100": score * 100.0,
                    })
            except Exception:
                err = traceback.format_exc(limit=5)
                for row in batch_rows:
                    failures.append({"stage": "inference", "image_path": row["image_path"], "error": err})
            print(f"Scored {len(predictions)}/{len(sampled_rows)}")
    except Exception:
        failures.append({"stage": "model_load", "image_path": "N/A", "error": traceback.format_exc(limit=10)})

    metrics_all = compute_aggregate_metrics(predictions)
    
    # Decision logic
    if not predictions:
        decision = "MDIQA_FLIVE_BENCHMARK_FAILED"
    elif metrics_all.get("srcc", 0) > PRODUCTION_FLIVE_SRCC:
        decision = "MDIQA_FLIVE_READY_FOR_FULL_IMAGE_BENCHMARK"
    elif metrics_all.get("srcc", 0) > 0.4:
        decision = "MDIQA_FLIVE_PROMISING_BUT_NEEDS_FULL_TEST"
    else:
        decision = "MDIQA_FLIVE_STILL_BELOW_PRODUCTION"

    elapsed = time.perf_counter() - start
    
    # Save files
    if predictions:
        with PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
            
        with METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_all.keys())
            writer.writeheader()
            writer.writerow(metrics_all)
            
        # Sensitivity checks
        p_arr = np.array([p["pred_raw"] for p in predictions])
        m_arr = np.array([p["quality_score_0_100"] for p in predictions])
        sensitivity = [
            {"check": "SRCC(pred, MOS)", "value": spearman(p_arr, m_arr)},
            {"check": "SRCC(-pred, MOS)", "value": spearman(-p_arr, m_arr)},
            {"check": "SRCC(pred, 100-MOS)", "value": spearman(p_arr, 100 - m_arr)},
        ]
        with SENSITIVITY_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["check", "value"])
            writer.writeheader()
            writer.writerows(sensitivity)

    if failures:
        with FAILURES_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["stage", "image_path", "error"])
            writer.writeheader()
            writer.writerows(failures)

    write_report(cuda, filter_rule, len(sampled_rows), predictions, failures, elapsed, decision)
    
    print(f"Final decision: {decision}")
    print(f"Report: {REPORT_MD}")

if __name__ == "__main__":
    main()
