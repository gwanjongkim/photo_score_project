# MDIQA 후보 기술 IQA teacher를 5장 이미지로 스모크 테스트한다.
from __future__ import annotations

import csv
import importlib
import os
import sys
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MDIQA_REPO = ROOT / "MDIQA"
OPEN_CLIP_REPO = ROOT / "open_clip"
OUTPUT_DIR = ROOT / "outputs" / "mdiqa_smoke_20260601"
PREDICTIONS_CSV = OUTPUT_DIR / "smoke_predictions.csv"
REPORT_MD = OUTPUT_DIR / "report.md"
TEST_SAMPLES_DIR = ROOT / "test_samples"
MANIFEST_CSV = (
    ROOT
    / "outputs"
    / "research_protocol_iqa_manifests_20260530"
    / "benchmark_all_research_protocol_iqa.csv"
)

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
class SmokeResult:
    image_path: str
    raw_score: float | None
    error: str = ""


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
        with MANIFEST_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = Path(row["image_path"])
                if path.exists() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    manifest_images.append(path)
                if len(manifest_images) == 5:
                    break
    if len(manifest_images) >= 5:
        return manifest_images
    raise RuntimeError("Could not find 5 local smoke-test images.")


def missing_checkpoints() -> list[Path]:
    return [path for path in CHECKPOINTS.values() if not path.exists() or path.stat().st_size < 1024]


def load_image_tensor(path: Path, size: int = 384) -> Any:
    from PIL import Image
    import numpy as np
    import torch

    image = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def force_timm_no_pretrained() -> None:
    import timm

    original_create_model = timm.create_model

    def create_model_no_pretrain(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = create_model_no_pretrain


class MDIQAKoniqSmokeModel:
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
            crop_size=384,
            semantic_model_name="resnet50",
            backbone_pretrain=False,
            pretrained=True,
            pretrained_model_path=technology_paths,
        ).to(self.device)
        self.aesthetic_model = CFANet_Shared(
            setting=2,
            num_class=4,
            num_crop=1,
            crop_size=384,
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
        state_dict = state.get("params", state)
        self.fusion_model.load_state_dict(state_dict, strict=True)
        self.technology_model.eval()
        self.aesthetic_model.eval()
        self.fusion_model.eval()

    def predict(self, image_path: Path) -> float:
        import torch

        tensor = load_image_tensor(image_path).to(self.device)
        with torch.no_grad():
            tech_global, tech_local, tech_tokens = self.technology_model(tensor)
            aes_global, aes_local, aes_tokens = self.aesthetic_model(tensor)
            global_scores = torch.cat([*tech_global, *aes_global], dim=1)
            local_scores = [*tech_local, *aes_local]
            final_tokens = [*tech_tokens, *aes_tokens]
            score = self.fusion_model(tensor, [global_scores, local_scores, final_tokens])
        return float(score.reshape(-1)[0].detach().cpu().item())


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


def write_report(
    decision: str,
    image_paths: list[Path],
    results: list[SmokeResult],
    checkpoint_status: list[str],
    failure: str = "",
) -> None:
    scores = [result.raw_score for result in results if result.raw_score is not None]
    scale = infer_scale(scores)
    lines = [
        "# MDIQA WSL Smoke Test Report",
        "",
        "## Scope",
        "- Smoke test only.",
        "- No training, TFLite export, Flutter edits, production scoring edits, distillation, or teacher-label creation.",
        "- License risk: S-Lab Non-commercial / Academic-only.",
        "",
        "## Environment",
        f"- environment_path: `{ROOT / '.venv_mdiqa'}`",
        f"- repo_path: `{MDIQA_REPO}`",
        f"- open_clip_path: `{OPEN_CLIP_REPO}`",
        "- device: `cpu` unless CUDA is visible to PyTorch.",
        "",
        "## Checkpoints",
        *[f"- {line}" for line in checkpoint_status],
        "",
        "## Images",
        *[f"- {path}" for path in image_paths],
        "",
        "## Raw Outputs",
    ]
    if results:
        for result in results:
            if result.error:
                lines.append(f"- {Path(result.image_path).name}: FAILED")
            else:
                lines.append(f"- {Path(result.image_path).name}: {result.raw_score}")
    else:
        lines.append("- No outputs.")
    lines.extend(["", "## Output Scale", f"- {scale}", "", "## Failures"])
    failed = [result for result in results if result.error]
    if failure:
        lines.extend(["```text", failure.strip(), "```"])
    elif failed:
        for result in failed:
            lines.extend([f"### {result.image_path}", "```text", result.error.strip(), "```"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Final Decision", decision, ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_local_caches()
    image_paths = select_images()

    checkpoint_status = []
    for name, path in CHECKPOINTS.items():
        status = "present" if path.exists() and path.stat().st_size >= 1024 else "missing"
        size = path.stat().st_size if path.exists() else 0
        checkpoint_status.append(f"{name}: `{path}` ({status}, {size} bytes)")

    missing = missing_checkpoints()
    if missing:
        decision = "MDIQA_BLOCKED_BY_CHECKPOINT_DOWNLOAD"
        write_csv_rows: list[dict[str, str]] = []
        write_predictions(write_csv_rows)
        write_report(
            decision,
            image_paths,
            [],
            checkpoint_status,
            "Missing checkpoint(s):\n" + "\n".join(str(path) for path in missing),
        )
        print(f"Final decision: {decision}")
        return

    results: list[SmokeResult] = []
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MDIQAKoniqSmokeModel(device=device)
        for path in image_paths:
            try:
                raw_score = model.predict(path)
                print(f"{path.name}: {raw_score}")
                results.append(SmokeResult(str(path), raw_score))
            except Exception:
                error = traceback.format_exc(limit=12)
                print(f"{path.name}: FAILED")
                print(error)
                results.append(SmokeResult(str(path), None, error))
    except Exception:
        failure = traceback.format_exc(limit=20)
        decision = "MDIQA_SMOKE_FAILED"
        write_predictions([])
        write_report(decision, image_paths, results, checkpoint_status, failure)
        print(failure)
        print(f"Final decision: {decision}")
        return

    rows = [
        {
            "image_path": result.image_path,
            "raw_score": "" if result.raw_score is None else repr(result.raw_score),
            "error": result.error,
        }
        for result in results
    ]
    write_predictions(rows)

    if any(result.error for result in results) or len([r for r in results if r.raw_score is not None]) != 5:
        decision = "MDIQA_SMOKE_FAILED"
    else:
        decision = "MDIQA_READY_FOR_100_IMAGE_SANITY"
    write_report(decision, image_paths, results, checkpoint_status)
    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Final decision: {decision}")


def write_predictions(rows: list[dict[str, str]]) -> None:
    with PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "raw_score", "error"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
