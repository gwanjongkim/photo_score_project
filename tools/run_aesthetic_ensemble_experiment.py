#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.infer.predict_quality_bundle import add_bundle_arguments
from src.infer.score_image_folder import score_folder_results, write_score_outputs
from src.infer.select_best_shots import normalize_unit_score, score_component


EXPERIMENT_NAME = "aesthetic_ensemble_rgnet_alamp_aadb_nima"
DEFAULT_MODEL_PATHS = {
    "stage5_student_model": "checkpoints/student_full_aadb/final_model.keras",
    "aadb_model": "checkpoints/composition_aadb_gpu/final_model.keras",
    "nima_model": "checkpoints/nima_ava_gpu/final_model.keras",
    "alamp_model": "checkpoints/alamp_aadb_gpu/final_model.keras",
    "rgnet_model": "checkpoints/rgnet_aadb_gpu/final_model.keras",
    "koniq_model": "checkpoints/technical_koniq_gpu/final_model.keras",
    "flive_image_model": "checkpoints/technical_flive_image_gpu/final_model.keras",
    "flive_patch_model": "checkpoints/technical_flive_patch_gpu/final_model.keras",
}
ENSEMBLE_WEIGHTS = {
    "rgnet_aadb": 0.30,
    "alamp_aadb": 0.30,
    "aadb_composition": 0.25,
    "nima_ava_unit": 0.15,
}
CURRENT_WEIGHTS_PATH = REPO_ROOT / "configs" / "stage5_student_server_weights.json"
CURRENT_SERVER_CONFIG_PATH = REPO_ROOT / "configs" / "stage5_student_server.json"
REFERENCE_CONFIG_PATH = REPO_ROOT / "configs" / "stage5_reference.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Non-production aesthetic ensemble experiment. Scores a folder with the existing "
            "stage5, AADB, NIMA, A-Lamp, RGNet, and technical bundle paths, then compares the "
            "current stage5 aesthetic path against an experimental weighted aesthetic ensemble."
        )
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.webp,.bmp")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--no_contact_sheets", action="store_true")
    return add_bundle_arguments(parser)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def apply_default_paths(args: argparse.Namespace) -> None:
    for field_name, default_path in DEFAULT_MODEL_PATHS.items():
        if getattr(args, field_name, None):
            continue
        resolved = resolve_project_path(default_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Required experiment model is missing for {field_name}: {resolved}")
        setattr(args, field_name, str(resolved))

    # Match the current repo defaults used by the individual model paths.
    if args.stage5_student_image_size is None:
        args.stage5_student_image_size = 224
    if args.image_size is None:
        args.image_size = 224
    if args.rgnet_image_size is None:
        args.rgnet_image_size = 256
    if args.alamp_global_size is None:
        args.alamp_global_size = 384
    if args.alamp_patch_size is None:
        args.alamp_patch_size = 224
    if args.alamp_num_patches is None:
        args.alamp_num_patches = 5
    if args.alamp_patch_scale is None:
        args.alamp_patch_scale = 0.35


def load_current_selector_weights() -> dict[str, Any]:
    if not CURRENT_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Current selector weights config not found: {CURRENT_WEIGHTS_PATH}")
    return json.loads(CURRENT_WEIGHTS_PATH.read_text(encoding="utf-8"))


def maybe_score(value: object) -> float | None:
    return normalize_unit_score(value)


def combine_final_score(aesthetic_score: float | None, technical_score: float | None, mix: dict[str, object]) -> float | None:
    if aesthetic_score is not None and technical_score is not None:
        return float(float(mix["aesthetic"]) * aesthetic_score + float(mix["technical"]) * technical_score)
    if aesthetic_score is not None:
        return float(aesthetic_score)
    if technical_score is not None:
        return float(technical_score)
    return None


def weighted_ensemble_for(record: dict[str, object]) -> tuple[float | None, dict[str, float | None]]:
    components = {
        "rgnet_aadb": maybe_score(record.get("rgnet_score")),
        "alamp_aadb": maybe_score(record.get("alamp_score")),
        "aadb_composition": maybe_score(record.get("aadb_score")),
        "nima_ava_unit": maybe_score(record.get("nima_mean_score")),
    }
    if any(value is None for value in components.values()):
        return None, components
    score = sum(float(ENSEMBLE_WEIGHTS[key]) * float(value) for key, value in components.items())
    return float(score), components


def compute_experiment_rows(results: list[dict[str, object]], weights: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        record = dict(result)
        stage5_aesthetic, stage5_contrib, stage5_missing = score_component(
            record,
            weights=weights["aesthetic"],
            component_name="aesthetic",
        )
        technical_component, technical_contrib, technical_missing = score_component(
            record,
            weights=weights["technical"],
            component_name="technical",
        )
        ensemble_score, ensemble_components = weighted_ensemble_for(record)
        stage5_final = combine_final_score(stage5_aesthetic, technical_component, weights["component_mix"])
        ensemble_final = combine_final_score(ensemble_score, technical_component, weights["component_mix"])

        row = {
            "image_path": record.get("image_path"),
            "stage5_aesthetic_score": stage5_aesthetic,
            "weighted_aesthetic_ensemble": ensemble_score,
            "aadb_composition": ensemble_components["aadb_composition"],
            "nima_mean_score_raw": None if record.get("nima_mean_score") is None else float(record["nima_mean_score"]),
            "nima_ava_unit": ensemble_components["nima_ava_unit"],
            "alamp_aadb": ensemble_components["alamp_aadb"],
            "rgnet_aadb": ensemble_components["rgnet_aadb"],
            "technical_component": technical_component,
            "stage5_current_final_score": stage5_final,
            "ensemble_experiment_final_score": ensemble_final,
            "stage5_minus_ensemble_aesthetic": (
                None if stage5_aesthetic is None or ensemble_score is None else float(stage5_aesthetic - ensemble_score)
            ),
            "stage5_minus_ensemble_final": (
                None if stage5_final is None or ensemble_final is None else float(stage5_final - ensemble_final)
            ),
            "koniq_score": record.get("koniq_score"),
            "flive_image_score": record.get("flive_image_score"),
            "flive_patch_mean": record.get("flive_patch_mean"),
            "flive_patch_min": record.get("flive_patch_min"),
            "stage5_aesthetic_contributions_json": json.dumps(stage5_contrib, ensure_ascii=False, sort_keys=True),
            "technical_contributions_json": json.dumps(technical_contrib, ensure_ascii=False, sort_keys=True),
            "missing_for_stage5_current": ",".join(sorted(stage5_missing)),
            "missing_for_technical": ",".join(sorted(technical_missing)),
            "ensemble_formula": "0.30*rgnet_aadb + 0.30*alamp_aadb + 0.25*aadb_composition + 0.15*nima_ava_unit",
            "normalization_notes": "AADB/A-Lamp/RGNet are clipped unit scores; NIMA uses (mean_score - 1.0) / 9.0.",
        }
        rows.append(row)
    return rows


def add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    rank_specs = {
        "rank_stage5_aesthetic": "stage5_aesthetic_score",
        "rank_ensemble_aesthetic": "weighted_aesthetic_ensemble",
        "rank_stage5_current_final": "stage5_current_final_score",
        "rank_ensemble_experiment_final": "ensemble_experiment_final_score",
        "rank_aadb_composition": "aadb_composition",
        "rank_nima_ava_unit": "nima_ava_unit",
        "rank_alamp_aadb": "alamp_aadb",
        "rank_rgnet_aadb": "rgnet_aadb",
    }
    for rank_col, score_col in rank_specs.items():
        ranked[rank_col] = ranked[score_col].rank(method="min", ascending=False, na_option="bottom").astype(int)

    ranked["aesthetic_rank_delta_ensemble_minus_stage5"] = (
        ranked["rank_ensemble_aesthetic"] - ranked["rank_stage5_aesthetic"]
    )
    ranked["final_rank_delta_ensemble_minus_stage5"] = (
        ranked["rank_ensemble_experiment_final"] - ranked["rank_stage5_current_final"]
    )
    ranked["abs_aesthetic_rank_delta"] = ranked["aesthetic_rank_delta_ensemble_minus_stage5"].abs()
    ranked["abs_final_rank_delta"] = ranked["final_rank_delta_ensemble_minus_stage5"].abs()
    return ranked


def summarize_score(frame: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "max": None, "range": None, "mean": None, "std": None}
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
    }


def top_records(frame: pd.DataFrame, score_col: str, limit: int) -> list[dict[str, object]]:
    columns = list(dict.fromkeys([
        "image_path",
        score_col,
        "stage5_aesthetic_score",
        "weighted_aesthetic_ensemble",
        "stage5_current_final_score",
        "ensemble_experiment_final_score",
        "rank_stage5_current_final",
        "rank_ensemble_experiment_final",
    ]))
    rows = frame.sort_values(score_col, ascending=False).head(limit)[columns]
    return json.loads(rows.to_json(orient="records"))


def overlap_summary(frame: pd.DataFrame, top_k: int) -> dict[str, object]:
    comparisons = [
        ("stage5_aesthetic_vs_ensemble_aesthetic", "stage5_aesthetic_score", "weighted_aesthetic_ensemble"),
        ("stage5_final_vs_ensemble_final", "stage5_current_final_score", "ensemble_experiment_final_score"),
    ]
    payload: dict[str, object] = {}
    for name, left_col, right_col in comparisons:
        left = set(frame.nlargest(top_k, left_col)["image_path"])
        right = set(frame.nlargest(top_k, right_col)["image_path"])
        payload[name] = {
            "top_k": int(top_k),
            "overlap_count": int(len(left & right)),
            "stage5_only": sorted(Path(path).name for path in left - right),
            "ensemble_only": sorted(Path(path).name for path in right - left),
            "overlap_images": sorted(Path(path).name for path in left & right),
        }
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def safe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


def make_contact_sheet(frame: pd.DataFrame, title: str, score_col: str, output_path: Path, limit: int = 10) -> None:
    rows = frame.head(limit).reset_index(drop=True)
    thumb_w, thumb_h = 240, 180
    label_h, title_h, pad = 60, 34, 8
    cols = 5
    grid_rows = math.ceil(len(rows) / cols)
    canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad, title_h + grid_rows * (thumb_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 8), title, fill="black", font=font)

    for index, row in rows.iterrows():
        image_path = Path(str(row["image_path"]))
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_w, thumb_h), "white")
        tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))

        col = index % cols
        grid_row = index // cols
        x = pad + col * (thumb_w + pad)
        y = title_h + grid_row * (thumb_h + label_h + pad)
        canvas.paste(tile, (x, y))
        label = (
            f"#{index + 1} {image_path.name}\n"
            f"{score_col}={float(row[score_col]):.3f}\n"
            f"s5={float(row['stage5_aesthetic_score']):.3f} ens={float(row['weighted_aesthetic_ensemble']):.3f}"
        )
        draw.text((x, y + thumb_h + 3), label, fill="black", font=font)

    canvas.save(output_path, quality=92)


def write_contact_sheets(frame: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    make_contact_sheet(
        frame.sort_values("stage5_current_final_score", ascending=False),
        "Top current stage5 final ranking",
        "stage5_current_final_score",
        output_dir / "contact_top_stage5_current_final.jpg",
        limit=top_k,
    )
    make_contact_sheet(
        frame.sort_values("ensemble_experiment_final_score", ascending=False),
        "Top experimental ensemble final ranking",
        "ensemble_experiment_final_score",
        output_dir / "contact_top_ensemble_experiment_final.jpg",
        limit=top_k,
    )
    make_contact_sheet(
        frame.sort_values("stage5_minus_ensemble_aesthetic", ascending=True),
        "Stage5 much lower than ensemble aesthetic",
        "stage5_minus_ensemble_aesthetic",
        output_dir / "contact_stage5_lower_than_ensemble.jpg",
        limit=top_k,
    )
    make_contact_sheet(
        frame.sort_values("abs_final_rank_delta", ascending=False),
        "Largest final-rank disagreements",
        "abs_final_rank_delta",
        output_dir / "contact_largest_final_rank_disagreements.jpg",
        limit=top_k,
    )


def write_outputs(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    weights: dict[str, Any],
    num_results: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values("rank_ensemble_experiment_final")
    stage5_ranked = frame.sort_values("rank_stage5_current_final")
    ensemble_ranked = frame.sort_values("rank_ensemble_experiment_final")
    disagreements = frame.sort_values(["abs_final_rank_delta", "abs_aesthetic_rank_delta"], ascending=False)
    stage5_lower = frame.sort_values("stage5_minus_ensemble_aesthetic", ascending=True)
    stage5_higher = frame.sort_values("stage5_minus_ensemble_aesthetic", ascending=False)

    ordered.to_csv(output_dir / "scores_with_ensemble.csv", index=False)
    stage5_ranked.to_csv(output_dir / "ranked_stage5_current.csv", index=False)
    ensemble_ranked.to_csv(output_dir / "ranked_ensemble_experiment.csv", index=False)
    disagreements.to_csv(output_dir / "disagreement_cases.csv", index=False)
    stage5_lower.to_csv(output_dir / "stage5_lower_than_ensemble.csv", index=False)
    stage5_higher.to_csv(output_dir / "stage5_higher_than_ensemble.csv", index=False)

    write_json(output_dir / "scores_with_ensemble.json", safe_records(ordered))
    write_json(output_dir / "ranked_stage5_current.json", safe_records(stage5_ranked))
    write_json(output_dir / "ranked_ensemble_experiment.json", safe_records(ensemble_ranked))
    write_json(output_dir / "disagreement_cases.json", safe_records(disagreements))

    score_columns = [
        "stage5_aesthetic_score",
        "weighted_aesthetic_ensemble",
        "stage5_current_final_score",
        "ensemble_experiment_final_score",
        "aadb_composition",
        "nima_ava_unit",
        "alamp_aadb",
        "rgnet_aadb",
        "technical_component",
    ]
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "non_production": True,
        "input_dir": str(resolve_project_path(args.input_dir)),
        "output_dir": str(output_dir),
        "num_scored_images": int(num_results),
        "formula": "0.30 * rgnet_aadb + 0.30 * alamp_aadb + 0.25 * aadb_composition + 0.15 * nima_ava_unit",
        "normalization": {
            "aadb_composition": "clip(aadb_score, 0, 1)",
            "alamp_aadb": "clip(alamp_score, 0, 1)",
            "rgnet_aadb": "clip(rgnet_score, 0, 1)",
            "nima_ava_unit": "clip((nima_mean_score - 1.0) / 9.0, 0, 1)",
            "technical_component": "src.infer.select_best_shots.normalize_unit_score with current stage5 server technical weights",
        },
        "model_paths": {field: getattr(args, field) for field in DEFAULT_MODEL_PATHS},
        "source_configs": {
            "current_server_config": str(CURRENT_SERVER_CONFIG_PATH),
            "current_server_weights": str(CURRENT_WEIGHTS_PATH),
            "reference_config_for_component_paths": str(REFERENCE_CONFIG_PATH),
        },
        "current_component_mix": weights["component_mix"],
        "score_summary": {column: summarize_score(frame, column) for column in score_columns},
        "top10_overlap": overlap_summary(frame, top_k=min(10, max(1, int(args.top_k)))),
        "top_stage5_current_final": top_records(frame, "stage5_current_final_score", int(args.top_k)),
        "top_ensemble_experiment_final": top_records(frame, "ensemble_experiment_final_score", int(args.top_k)),
        "largest_final_rank_disagreements": safe_records(disagreements.head(int(args.top_k))),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "experiment_config.json",
        {
            "experiment_name": EXPERIMENT_NAME,
            "non_production": True,
            "formula": summary["formula"],
            "ensemble_weights": ENSEMBLE_WEIGHTS,
            "defaults": DEFAULT_MODEL_PATHS,
            "cli_args": vars(args),
        },
    )
    write_json(output_dir / "top10_overlap.json", summary["top10_overlap"])
    pd.DataFrame(
        [
            {"comparison": name, **payload}
            for name, payload in summary["top10_overlap"].items()
        ]
    ).to_csv(output_dir / "top10_overlap.csv", index=False)

    if not args.no_contact_sheets:
        write_contact_sheets(frame, output_dir=output_dir, top_k=min(10, max(1, int(args.top_k))))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_default_paths(args)

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / "raw_bundle_scores.jsonl"
    raw_csv = output_dir / "raw_bundle_scores.csv"

    weights = load_current_selector_weights()
    results = score_folder_results(args)
    write_score_outputs(results, output_jsonl=raw_jsonl, output_csv=raw_csv)

    rows = compute_experiment_rows(results, weights=weights)
    frame = add_ranks(pd.DataFrame(rows))
    write_outputs(frame=frame, output_dir=output_dir, args=args, weights=weights, num_results=len(results))

    print(
        json.dumps(
            {
                "experiment_name": EXPERIMENT_NAME,
                "output_dir": str(output_dir),
                "num_scored_images": int(len(results)),
                "scores_csv": str(output_dir / "scores_with_ensemble.csv"),
                "summary_json": str(output_dir / "summary.json"),
            },
            indent=args.json_indent,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
