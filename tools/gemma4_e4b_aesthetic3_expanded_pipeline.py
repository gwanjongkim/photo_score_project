from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PRIMARY_WEIGHTS = {
    "nima_unit_score": 0.10,
    "rgnet_score": 0.40,
    "alamp_score": 0.50,
}
COMPARISON_WEIGHTS = {
    "nima_unit_score": 0.15,
    "rgnet_score": 0.45,
    "alamp_score": 0.40,
}
FORBIDDEN_COMMENT_RE = re.compile(
    r"\b(score|scores|threshold|weighted|rank|ranking|ranked|top[- ]?k|percentile|"
    r"nima|rgnet|a-lamp|alamp|model|technical|composition score|selection_state)\b|"
    r"\b\d+(?:\.\d+)?\b",
    flags=re.IGNORECASE,
)
GENERIC_COMMENT_RE = re.compile(
    r"\b(overall appeal|aesthetic appeal|clear visual strengths|usable strengths|"
    r"weaker secondary cues|higher-ranked|selected frame|stronger alternatives|"
    r"competitive visual evidence|weaker a-cut impact|clear subject emphasis|"
    r"weak subject separation|a distracting background)\b",
    flags=re.IGNORECASE,
)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def command_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def repo_rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def resolve_image_path(value: object) -> Path:
    text = "" if value is None else str(value).strip()
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def image_key(path: str | Path) -> str:
    return str(resolve_image_path(path))


def stable_id(*parts: object) -> str:
    text = "__".join(str(part) for part in parts)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{text[:100]}__{digest}"


def normalize_unit_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    if -0.05 <= numeric <= 1.05:
        return float(np.clip(numeric, 0.0, 1.0))
    if 1.0 <= numeric <= 10.5:
        return float(np.clip((numeric - 1.0) / 9.0, 0.0, 1.0))
    if 0.0 <= numeric <= 100.0:
        return float(np.clip(numeric / 100.0, 0.0, 1.0))
    return None


def round_float(value: object, digits: int = 6) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return round(numeric, digits)


def validate_image(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle)
            width, height = image.size
            image.convert("RGB").load()
        if width < 2 or height < 2:
            return {"ok": False, "error": f"too_small:{width}x{height}"}
        return {"ok": True, "width": int(width), "height": int(height)}
    except Exception as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}


def csv_exists_summary(paths: list[Path]) -> list[dict[str, Any]]:
    return [{"path": repo_rel(path), "exists": path.exists(), "rows": int(len(pd.read_csv(path))) if path.exists() else 0} for path in paths]


def lightweight_path_verification() -> dict[str, Any]:
    ava_csvs = [repo_path(f"data/processed/ava/{split}.csv") for split in ("train", "val", "test")]
    aadb_csvs = [repo_path(f"data/processed/aadb/{split}.csv") for split in ("train", "val", "test")]
    test_vila_dir = repo_path("test_vila")
    sample_results: list[dict[str, Any]] = []
    for dataset_name, split_paths in (("ava", ava_csvs), ("aadb", aadb_csvs)):
        for csv_path in split_paths:
            if not csv_path.exists():
                sample_results.append({"dataset": dataset_name, "csv": repo_rel(csv_path), "ok": False, "error": "missing_csv"})
                continue
            frame = pd.read_csv(csv_path, nrows=5)
            for row in frame.to_dict(orient="records")[:3]:
                path = resolve_image_path(row.get("image_path"))
                checked = validate_image(path)
                sample_results.append(
                    {
                        "dataset": dataset_name,
                        "csv": repo_rel(csv_path),
                        "image_path": str(path),
                        **checked,
                    }
                )

    test_vila_images = sorted(
        path for path in test_vila_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ) if test_vila_dir.exists() else []
    for path in test_vila_images[:5]:
        sample_results.append({"dataset": "test_vila", "image_path": str(path), **validate_image(path)})

    return {
        "ava_csvs": csv_exists_summary(ava_csvs),
        "aadb_csvs": csv_exists_summary(aadb_csvs),
        "ava_raw_images_path": str(repo_path("data/raw/ava/images")),
        "ava_raw_images_exists": repo_path("data/raw/ava/images").exists(),
        "aadb_raw_images_path": str(repo_path("data/raw/aadb/images")),
        "aadb_raw_images_exists": repo_path("data/raw/aadb/images").exists(),
        "test_vila_path": str(test_vila_dir),
        "test_vila_exists": test_vila_dir.exists(),
        "test_vila_image_count": len(test_vila_images),
        "sample_image_checks": sample_results,
        "all_lightweight_checks_ok": all(
            item.get("exists", True) and item.get("ok", True)
            for group in (csv_exists_summary(ava_csvs), csv_exists_summary(aadb_csvs), sample_results)
            for item in (group if isinstance(group, list) else [group])
        ) and test_vila_dir.exists(),
    }


def allocation_for_target(target: int) -> dict[str, int]:
    raw = {"train": int(round(target * 0.80)), "val": int(round(target * 0.10))}
    raw["test"] = target - raw["train"] - raw["val"]
    return raw


def load_dataset_rows(dataset_name: str, split: str) -> pd.DataFrame:
    path = repo_path(f"data/processed/{dataset_name}/{split}.csv")
    frame = pd.read_csv(path)
    frame["dataset_source"] = dataset_name
    frame["source_split"] = split
    frame["source_metadata_csv"] = repo_rel(path)
    frame["source_image_path"] = frame["image_path"].astype(str)
    frame["image_path"] = frame["image_path"].map(lambda value: str(resolve_image_path(value)))
    return frame


def sample_valid_rows(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    split: str,
    target: int,
    seed: int,
    seen_paths: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng_seed = seed + sum(ord(ch) for ch in f"{dataset_name}:{split}")
    shuffled = frame.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in shuffled.to_dict(orient="records"):
        path = str(row["image_path"])
        if path in seen_paths:
            continue
        checked = validate_image(Path(path))
        if not checked["ok"]:
            rejected.append({"dataset_source": dataset_name, "source_split": split, "image_path": path, "error": checked.get("error")})
            continue
        seen_paths.add(path)
        keep = {
            "id": stable_id(dataset_name, split, Path(path).name),
            "dataset_source": dataset_name,
            "source_split": split,
            "source_metadata_csv": row.get("source_metadata_csv"),
            "source_image_path": row.get("source_image_path"),
            "image_path": path,
            "image_file": Path(path).name,
            "width": checked.get("width"),
            "height": checked.get("height"),
        }
        if dataset_name == "ava":
            keep["ava_mean_score"] = round_float(row.get("mean_score"))
            for index in range(1, 11):
                if f"vote_{index}" in row:
                    keep[f"ava_vote_{index}"] = int(row.get(f"vote_{index}") or 0)
                if f"dist_{index}" in row:
                    keep[f"ava_dist_{index}"] = round_float(row.get(f"dist_{index}"))
        elif dataset_name == "aadb":
            keep["aadb_score"] = round_float(row.get("score"))
            keep["aadb_relative_path"] = row.get("relative_path")
        rows.append(keep)
        if len(rows) >= target:
            break
    return rows, rejected


def build_pool(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    verification = lightweight_path_verification()
    write_json(output_dir / "dataset_path_verification.json", verification)

    seen_paths: set[str] = set()
    pool_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    targets = {"ava": int(args.ava_target), "aadb": int(args.aadb_target)}
    created_counts: dict[str, dict[str, int]] = {}
    for dataset_name, target in targets.items():
        split_targets = allocation_for_target(target)
        created_counts[dataset_name] = {}
        for split, split_target in split_targets.items():
            frame = load_dataset_rows(dataset_name, split)
            rows, rejected = sample_valid_rows(
                frame,
                dataset_name=dataset_name,
                split=split,
                target=split_target,
                seed=int(args.seed),
                seen_paths=seen_paths,
            )
            pool_rows.extend(rows)
            rejected_rows.extend(rejected)
            created_counts[dataset_name][split] = len(rows)

    pool_rows.sort(key=lambda row: (row["dataset_source"], row["source_split"], row["image_path"]))
    pool_frame = pd.DataFrame(pool_rows)
    pool_frame.to_csv(output_dir / "image_pool.csv", index=False)
    write_jsonl(output_dir / "image_pool.jsonl", pool_rows)
    if rejected_rows:
        pd.DataFrame(rejected_rows).to_csv(output_dir / "image_pool_rejected_decode_failures.csv", index=False)
        write_jsonl(output_dir / "image_pool_rejected_decode_failures.jsonl", rejected_rows)

    summary = {
        "output_dir": repo_rel(output_dir),
        "requested_targets": targets,
        "created_counts": created_counts,
        "total_rows": len(pool_rows),
        "dataset_counts": {str(key): int(value) for key, value in pool_frame["dataset_source"].value_counts().to_dict().items()},
        "split_counts": {str(key): int(value) for key, value in pool_frame["source_split"].value_counts().to_dict().items()},
        "decode_rejected_rows": len(rejected_rows),
        "duplicates_avoided": "deduplicated_by_resolved_image_path",
    }
    write_json(output_dir / "image_pool_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def score_pool(args: argparse.Namespace) -> None:
    from argparse import Namespace

    from src.infer.predict_quality_bundle import load_bundle_models, predict_bundle_for_image

    output_dir = repo_path(args.output_dir)
    pool_path = output_dir / "image_pool.csv"
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing image pool: {pool_path}")
    pool_rows = pd.read_csv(pool_path).replace({np.nan: None}).to_dict(orient="records")
    progress_path = output_dir / "scoring_progress.jsonl"
    failure_path = output_dir / "score_failures.jsonl"
    existing = {str(row.get("image_path")): row for row in read_jsonl(progress_path) if row.get("image_path")}
    existing_failures = {str(row.get("image_path")): row for row in read_jsonl(failure_path) if row.get("image_path")}

    bundle_args = Namespace(
        stage5_student_model=None,
        stage5_student_image_size=224,
        aadb_model=None,
        koniq_model=None,
        flive_image_model=None,
        flive_patch_model=None,
        nima_model=str(repo_path(args.nima_model)),
        alamp_model=str(repo_path(args.alamp_model)),
        musiq_model=None,
        rgnet_model=str(repo_path(args.rgnet_model)),
        pairwise_model=None,
        pairwise_reference_csv=None,
        pairwise_target_col="score",
        pairwise_max_references=16,
        image_size=int(args.image_size),
        rgnet_image_size=int(args.rgnet_image_size),
        alamp_global_size=int(args.alamp_global_size),
        alamp_patch_size=int(args.alamp_patch_size),
        alamp_num_patches=int(args.alamp_num_patches),
        alamp_patch_scale=float(args.alamp_patch_scale),
        alamp_include_debug=False,
        include_debug=False,
        musiq_patch_size=32,
        musiq_scale_sizes="224,384,512",
        musiq_patches_per_scale=16,
        _auto_resolved_model_fields=[],
        _model_load_warnings=[],
    )
    models = load_bundle_models(bundle_args)

    pending = [
        row for row in pool_rows
        if str(row.get("image_path")) not in existing and str(row.get("image_path")) not in existing_failures
    ]
    if args.max_rows is not None:
        pending = pending[: int(args.max_rows)]
    for index, row in enumerate(pending, start=1):
        path = str(row["image_path"])
        try:
            result = predict_bundle_for_image(path, args=bundle_args, models=models)
            nima_unit = normalize_unit_score(result.get("nima_mean_score"))
            rgnet_score = normalize_unit_score(result.get("rgnet_score"))
            alamp_score = normalize_unit_score(result.get("alamp_score"))
            if None in (nima_unit, rgnet_score, alamp_score):
                raise ValueError(
                    f"missing requested score(s): nima={result.get('nima_mean_score')} rgnet={result.get('rgnet_score')} alamp={result.get('alamp_score')}"
                )
            payload = {
                "image_path": path,
                "nima_mean_score": round_float(result.get("nima_mean_score")),
                "nima_unit_score": round_float(nima_unit),
                "rgnet_score": round_float(rgnet_score),
                "alamp_score": round_float(alamp_score),
                "available_models": sorted((result.get("per_model") or {}).keys()),
            }
            if result.get("model_stage_failures"):
                payload["model_stage_failures"] = result.get("model_stage_failures")
            append_jsonl(progress_path, payload)
            existing[path] = payload
        except Exception as exc:
            failure = {"image_path": path, "error": f"{exc.__class__.__name__}: {exc}"}
            append_jsonl(failure_path, failure)
            existing_failures[path] = failure
        if index % int(args.progress_every) == 0 or index == len(pending):
            print(json.dumps({"phase": "score_progress", "processed_this_run": index, "pending_this_run": len(pending)}, ensure_ascii=False))

    finalize_scores(output_dir)


def weighted_score(row: dict[str, Any], weights: dict[str, float]) -> float | None:
    values = {key: normalize_unit_score(row.get(key)) for key in weights}
    if any(value is None for value in values.values()):
        return None
    return round_float(sum(float(weights[key]) * float(values[key]) for key in weights))


def finalize_scores(output_dir: Path) -> None:
    pool_path = output_dir / "image_pool.csv"
    pool_frame = pd.read_csv(pool_path).replace({np.nan: None})
    score_rows = read_jsonl(output_dir / "scoring_progress.jsonl")
    score_frame = pd.DataFrame(score_rows).drop_duplicates("image_path", keep="last") if score_rows else pd.DataFrame()
    if score_frame.empty:
        write_json(output_dir / "score_summary.json", {"rows": 0, "status": "no_scores"})
        return
    merged = pool_frame.merge(score_frame, on="image_path", how="inner")
    for col in ("nima_unit_score", "rgnet_score", "alamp_score"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=["nima_unit_score", "rgnet_score", "alamp_score"]).copy()
    merged["weighted_aesthetic_score"] = merged.apply(lambda row: weighted_score(row.to_dict(), PRIMARY_WEIGHTS), axis=1)
    merged["balanced_comparison_score"] = merged.apply(lambda row: weighted_score(row.to_dict(), COMPARISON_WEIGHTS), axis=1)
    merged["score_delta_primary_minus_comparison"] = (
        pd.to_numeric(merged["weighted_aesthetic_score"], errors="coerce")
        - pd.to_numeric(merged["balanced_comparison_score"], errors="coerce")
    ).round(6)
    merged = merged.sort_values(["weighted_aesthetic_score", "image_path"], ascending=[False, True]).reset_index(drop=True)
    merged["rank_primary"] = np.arange(1, len(merged) + 1)
    comp_rank = merged.sort_values(["balanced_comparison_score", "image_path"], ascending=[False, True]).reset_index()
    comp_rank["rank_comparison"] = np.arange(1, len(comp_rank) + 1)
    merged = merged.merge(comp_rank[["index", "rank_comparison"]], left_index=True, right_on="index", how="left").drop(columns=["index"])
    merged["rank_delta_primary_minus_comparison"] = merged["rank_primary"].astype(int) - merged["rank_comparison"].astype(int)
    total = max(1, len(merged))
    merged["primary_rank_percentile"] = merged["rank_primary"].map(lambda rank: (int(rank) - 1) / total)
    merged["selection_bucket"] = merged["primary_rank_percentile"].map(
        lambda pct: "selected_high" if pct < 0.30 else ("middle_near_miss" if pct < 0.70 else "rejected_low")
    )
    merged["selected"] = merged["selection_bucket"].eq("selected_high")

    scored_csv = output_dir / "scored_pool.csv"
    scored_jsonl = output_dir / "scored_pool.jsonl"
    merged.to_csv(scored_csv, index=False)
    write_jsonl(scored_jsonl, merged.replace({np.nan: None}).to_dict(orient="records"))

    numeric_summary: dict[str, Any] = {}
    for col in ("nima_unit_score", "rgnet_score", "alamp_score", "weighted_aesthetic_score", "balanced_comparison_score", "score_delta_primary_minus_comparison"):
        series = pd.to_numeric(merged[col], errors="coerce")
        numeric_summary[col] = {
            "min": round_float(series.min()),
            "mean": round_float(series.mean()),
            "median": round_float(series.median()),
            "max": round_float(series.max()),
        }

    summary = {
        "status": "ok",
        "pool_rows": int(len(pool_frame)),
        "scored_rows": int(len(merged)),
        "score_failures": int(len(read_jsonl(output_dir / "score_failures.jsonl"))),
        "primary_weight_formula": "weighted_aesthetic_score = 0.10 * nima_unit_score + 0.40 * rgnet_score + 0.50 * alamp_score",
        "comparison_weight_formula": "balanced_comparison_score = 0.15 * nima_unit_score + 0.45 * rgnet_score + 0.40 * alamp_score",
        "dataset_counts": {str(key): int(value) for key, value in merged["dataset_source"].value_counts().to_dict().items()},
        "bucket_counts": {str(key): int(value) for key, value in merged["selection_bucket"].value_counts().to_dict().items()},
        "numeric_summary": numeric_summary,
        "outputs": {
            "scored_pool_csv": repo_rel(scored_csv),
            "scored_pool_jsonl": repo_rel(scored_jsonl),
        },
    }
    write_json(output_dir / "score_summary.json", summary)

    rank_abs = merged["rank_delta_primary_minus_comparison"].abs()
    comparison_summary = {
        "primary_weight_formula": summary["primary_weight_formula"],
        "comparison_weight_formula": summary["comparison_weight_formula"],
        "mean_abs_rank_delta": round_float(rank_abs.mean()),
        "median_abs_rank_delta": round_float(rank_abs.median()),
        "max_abs_rank_delta": int(rank_abs.max()) if len(rank_abs) else 0,
        "large_rank_disagreement_rows_abs_delta_ge_100": int((rank_abs >= 100).sum()),
        "primary_minus_comparison_delta_summary": numeric_summary["score_delta_primary_minus_comparison"],
        "top_large_disagreements": merged.reindex(rank_abs.sort_values(ascending=False).index)
        .head(20)[["image_path", "dataset_source", "weighted_aesthetic_score", "balanced_comparison_score", "score_delta_primary_minus_comparison", "rank_primary", "rank_comparison", "rank_delta_primary_minus_comparison"]]
        .replace({np.nan: None})
        .to_dict(orient="records"),
    }
    write_json(output_dir / "comparison_weight_summary.json", comparison_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def image_uri(path: str) -> str:
    try:
        return Path(path).resolve(strict=False).as_uri()
    except ValueError:
        return html.escape(path)


def render_table(rows: list[dict[str, Any]], title: str, limit: int | None = None) -> str:
    visible = rows[:limit] if limit is not None else rows
    out = [f"<h2>{html.escape(title)} <span>{len(rows)} rows</span></h2>", "<div class='grid'>"]
    for row in visible:
        out.append(
            "<article class='card'>"
            f"<img src='{html.escape(image_uri(str(row['image_path'])))}' loading='lazy'>"
            f"<div class='meta'><b>#{int(row['rank_primary'])}</b> / cmp #{int(row['rank_comparison'])}</div>"
            f"<div class='file'>{html.escape(str(row['image_file']))}</div>"
            f"<div>{html.escape(str(row['dataset_source']))} / {html.escape(str(row['source_split']))}</div>"
            f"<div>NIMA {float(row['nima_unit_score']):.3f} | RGNet {float(row['rgnet_score']):.3f} | A-Lamp {float(row['alamp_score']):.3f}</div>"
            f"<div>Primary {float(row['weighted_aesthetic_score']):.3f} | Comparison {float(row['balanced_comparison_score']):.3f}</div>"
            f"<div>Delta {float(row['score_delta_primary_minus_comparison']):+.4f} | Rank delta {int(row['rank_delta_primary_minus_comparison']):+d}</div>"
            "</article>"
        )
    out.append("</div>")
    return "\n".join(out)


def build_review(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    scored_path = output_dir / "scored_pool.csv"
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing scored pool: {scored_path}")
    frame = pd.read_csv(scored_path)
    rows = frame.replace({np.nan: None}).to_dict(orient="records")
    high = [row for row in rows if row["selection_bucket"] == "selected_high"]
    middle = [row for row in rows if row["selection_bucket"] == "middle_near_miss"]
    low = [row for row in rows if row["selection_bucket"] == "rejected_low"]
    disagreement = sorted(rows, key=lambda row: abs(int(row["rank_delta_primary_minus_comparison"])), reverse=True)
    css = """
body{font-family:Georgia,'Times New Roman',serif;margin:0;background:#f4efe4;color:#1f2522}
header{padding:28px 34px;background:linear-gradient(120deg,#203a34,#bb8a52);color:#fff}
h1{margin:0;font-size:34px;letter-spacing:-.02em} h2{margin:34px 28px 14px;font-size:24px} h2 span{font-size:14px;color:#68716a}
.note{max-width:980px;line-height:1.45}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px;padding:0 28px 34px}
.card{background:#fffaf0;border:1px solid #d7c5a5;border-radius:16px;box-shadow:0 8px 20px #0001;overflow:hidden}
.card img{width:100%;height:180px;object-fit:cover;background:#ddd}.card div{padding:0 12px 8px}.card .meta{padding-top:10px;color:#6f4e28}
.file{font-size:13px;word-break:break-all;color:#29322d}
"""
    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Gemma 4 E4B Aesthetic3 Review</title>",
        f"<style>{css}</style></head><body>",
        "<header><h1>Gemma 4 E4B Aesthetic3 Expanded Review</h1>",
        "<p class='note'>Primary weight: 0.10 NIMA unit + 0.40 RGNet + 0.50 A-Lamp. Comparison weight: 0.15 NIMA unit + 0.45 RGNet + 0.40 A-Lamp. The disagreement section surfaces rows most affected by the heavier A-Lamp primary weight.</p></header>",
        render_table(high, "Selected / High", limit=int(args.section_limit)),
        render_table(middle, "Middle / Near Miss", limit=int(args.section_limit)),
        render_table(low, "Rejected / Low", limit=int(args.section_limit)),
        render_table(disagreement, "Large Rank Disagreement", limit=int(args.disagreement_limit)),
        "</body></html>",
    ]
    review_path = output_dir / "review.html"
    review_path.write_text("\n".join(body), encoding="utf-8")
    print(json.dumps({"review_html": repo_rel(review_path), "rows": len(rows)}, indent=2, ensure_ascii=False))


def source_balanced_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.head(0)
    desired = {"ava": int(round(count * 2 / 3))}
    desired["aadb"] = count - desired["ava"]
    samples = []
    remainder = count
    for source, target in desired.items():
        source_frame = frame[frame["dataset_source"] == source]
        take = min(target, len(source_frame))
        if take > 0:
            samples.append(source_frame.sample(n=take, random_state=seed + len(samples)))
            remainder -= take
    if remainder > 0:
        used_indices = set(pd.concat(samples).index) if samples else set()
        leftover = frame[~frame.index.isin(used_indices)]
        take = min(remainder, len(leftover))
        if take > 0:
            samples.append(leftover.sample(n=take, random_state=seed + 99))
    return pd.concat(samples).sample(frac=1.0, random_state=seed + 7).reset_index(drop=True) if samples else frame.head(0)


def build_teacher_subset(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    frame = pd.read_csv(output_dir / "scored_pool.csv")
    target = int(args.teacher_target)
    counts = {
        "selected_high": int(round(target * 0.30)),
        "middle_near_miss": int(round(target * 0.40)),
    }
    counts["rejected_low"] = target - counts["selected_high"] - counts["middle_near_miss"]
    selected_frames = []
    for bucket, count in counts.items():
        bucket_frame = frame[frame["selection_bucket"] == bucket].copy()
        selected_frames.append(source_balanced_sample(bucket_frame, count=count, seed=int(args.seed) + len(selected_frames)))
    subset = pd.concat(selected_frames).sample(frac=1.0, random_state=int(args.seed) + 100).reset_index(drop=True)
    subset["teacher_subset_bucket"] = subset["selection_bucket"]
    subset["selected"] = subset["teacher_subset_bucket"].eq("selected_high")
    subset["teacher_subset_id"] = [stable_id("teacher", index, row["dataset_source"], Path(str(row["image_path"])).name) for index, row in subset.iterrows()]
    subset.to_csv(output_dir / "teacher_subset_manifest.csv", index=False)
    rows = subset.replace({np.nan: None}).to_dict(orient="records")
    write_jsonl(output_dir / "teacher_subset_manifest.jsonl", rows)
    summary = {
        "requested_teacher_subset_size": target,
        "teacher_subset_size": int(len(subset)),
        "bucket_counts": {str(key): int(value) for key, value in subset["teacher_subset_bucket"].value_counts().to_dict().items()},
        "dataset_counts": {str(key): int(value) for key, value in subset["dataset_source"].value_counts().to_dict().items()},
        "selected_counts": {"selected": int(subset["selected"].sum()), "not_selected": int((~subset["selected"].astype(bool)).sum())},
        "test_vila_policy": "excluded_from_main_teacher_subset; used only as sanity/evaluation manifest",
    }
    write_json(output_dir / "teacher_subset_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def strict_teacher_prompt(row: dict[str, Any]) -> str:
    selected = bool(row.get("selected"))
    selection_state = "selected" if selected else "rejected"
    return f"""
You are judging whether this photo is useful for A-cut photo selection.
This is not a captioning task.
Use visible evidence first. The hidden selection state only controls whether the final judgment is selected or rejected.
Do not mention scores, ranks, thresholds, model names, datasets, file names, or hidden context.

Return exactly one valid JSON object and nothing else.
Use only these keys: short_reason, detailed_reason, comparison_reason, tags, confidence.

Rules:
- short_reason: one short clause that starts with "{selection_state}:"
- After "{selection_state}:", mention one visible strength and one visible limitation when possible
- detailed_reason: exactly two short sentences grounded in visible evidence
- comparison_reason: null unless a visible tradeoff is obvious from the image alone
- tags: JSON array with zero to three labels chosen from composition, clarity, subject_separation, lighting, background, distraction, blur, exposure
- confidence: null if unsure
- Never write digits or numeric ratings inside reason strings
- Do not use words like score, rank, threshold, model, top, percent, file, or dataset inside reason strings

Good short_reason examples:
- "{selection_state}: clean subject emphasis, busier background"
- "{selection_state}: strong light, limited separation"
- "{selection_state}: clear texture, weaker framing"

Hidden selection state: {selection_state}
""".strip()


def load_teacher_subset_rows(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "teacher_subset_manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing teacher subset manifest: {path}")
    return read_jsonl(path)


def validate_three_line(text: object, selected: bool | None = None) -> dict[str, Any]:
    raw = "" if text is None else str(text).replace("\r\n", "\n").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    prefix_ok = len(lines) == 3 and lines[0].startswith("Strength:") and lines[1].startswith("Weakness:") and lines[2].startswith("Verdict:")
    values = [re.sub(r"^[A-Za-z]+:\s*", "", line).strip().lower() for line in lines[:3]]
    role_ok = prefix_ok and len(set(values)) == 3
    polarity_ok = True
    if prefix_ok and selected is not None:
        expected = "Verdict: selected because" if selected else "Verdict: not selected because"
        polarity_ok = lines[2].lower().startswith(expected.lower())
    leakage = bool(FORBIDDEN_COMMENT_RE.search(raw))
    generic = bool(GENERIC_COMMENT_RE.search(raw))
    return {
        "line_count": len(lines),
        "prefix_ok": prefix_ok,
        "role_separation_ok": role_ok,
        "verdict_polarity_ok": polarity_ok,
        "leakage": leakage,
        "generic": generic,
        "is_clean": prefix_ok and role_ok and polarity_ok and not leakage and not generic,
        "lines": lines,
    }


def sanitize_phrase(value: str, fallback: str) -> str:
    text = re.sub(r"\b(score|scores|threshold|weighted|rank|ranking|ranked|top[- ]?k|percentile|nima|rgnet|a-lamp|alamp|model|technical|selection_state)\b", "", value, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;-")
    return text or fallback


def sentence(value: str) -> str:
    text = value.strip()
    if not text:
        return text
    text = text[0].lower() + text[1:] if text[:1].isupper() and len(text) > 1 else text
    if text[-1] not in ".!?":
        text += "."
    return text


def repair_comment(text: object, selected: bool) -> str:
    raw = "" if text is None else str(text).replace("\r\n", "\n").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    strength = ""
    weakness = ""
    if lines:
        strength = re.sub(r"^Strength:\s*", "", lines[0], flags=re.IGNORECASE)
    if len(lines) > 1:
        weakness = re.sub(r"^Weakness:\s*", "", lines[1], flags=re.IGNORECASE)
    strength = sanitize_phrase(strength, "clear subject emphasis")
    weakness = sanitize_phrase(weakness, "a distracting background" if selected else "weak subject separation")
    verdict_prefix = "selected because" if selected else "not selected because"
    if selected:
        verdict = f"{verdict_prefix} {strength.strip(' .')} carries the frame despite {weakness.strip(' .')}"
    else:
        verdict = f"{verdict_prefix} {weakness.strip(' .')} undercuts {strength.strip(' .')}"
    return "\n".join([f"Strength: {sentence(strength)}", f"Weakness: {sentence(weakness)}", f"Verdict: {sentence(verdict)}"])


def worker_row_to_comment(worker_row: dict[str, Any], selected: bool) -> str:
    from src.gemma_distill.teacher import format_teacher_comment_3line

    raw_text = worker_row.get("raw_text") or worker_row.get("normalized_text")
    validation = validate_three_line(raw_text, selected=selected)
    if validation["prefix_ok"]:
        return "\n".join(validation["lines"])
    raw_text_value = "" if raw_text is None else str(raw_text)

    def raw_field(field_name: str) -> str | None:
        match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"([^"]+)"', raw_text_value, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    short_reason = worker_row.get("short_reason") or raw_field("short_reason")
    detailed_reason = worker_row.get("detailed_reason") or raw_field("detailed_reason")
    comparison_reason = worker_row.get("comparison_reason") or raw_field("comparison_reason")
    return format_teacher_comment_3line(
        short_reason=short_reason,
        detailed_reason=detailed_reason,
        comparison_reason=comparison_reason,
        selected=selected,
    )


def generate_teacher(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    rows = load_teacher_subset_rows(output_dir)
    if args.max_rows is not None:
        allowed_ids = {str(row["teacher_subset_id"]) for row in rows[: int(args.max_rows)]}
    else:
        allowed_ids = {str(row["teacher_subset_id"]) for row in rows}

    progress_path = output_dir / "teacher_generation_progress.jsonl"
    existing = {str(row.get("teacher_subset_id")): row for row in read_jsonl(progress_path) if row.get("teacher_subset_id")}
    pending = [row for row in rows if str(row["teacher_subset_id"]) in allowed_ids and str(row["teacher_subset_id"]) not in existing]
    chunk_dir = output_dir / "teacher_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    vila_python = command_path(args.vila_python)
    model_path = repo_path(args.vila_model_path)

    for chunk_start in range(0, len(pending), int(args.chunk_size)):
        chunk = pending[chunk_start: chunk_start + int(args.chunk_size)]
        if not chunk:
            break
        chunk_id = f"chunk_{chunk_start:05d}_{chunk_start + len(chunk) - 1:05d}"
        input_path = chunk_dir / f"{chunk_id}_input.json"
        stdout_path = chunk_dir / f"{chunk_id}_stdout.txt"
        output_path = chunk_dir / f"{chunk_id}_output.json"
        payload = {"rows": [{"image_path": row["image_path"], "prompt": strict_teacher_prompt(row)} for row in chunk]}
        write_json(input_path, payload)
        command = [
            str(vila_python),
            "-m",
            "src.infer.vila_full_local_worker",
            "--model_path",
            str(model_path),
            "--input_json",
            str(input_path),
            "--device",
            str(args.device),
            "--max_new_tokens",
            str(int(args.max_new_tokens)),
        ]
        print(json.dumps({"phase": "teacher_chunk_start", "chunk": chunk_id, "rows": len(chunk), "command": command}, ensure_ascii=False))
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=False)
        stdout_path.write_text((completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
        if completed.returncode != 0:
            write_json(output_dir / "teacher_generation_blocker.json", {"chunk": chunk_id, "returncode": completed.returncode, "stdout_path": repo_rel(stdout_path)})
            raise RuntimeError(f"VILA worker failed for {chunk_id}; see {stdout_path}")
        text = completed.stdout.strip()
        start = text.find("{")
        if start < 0:
            write_json(output_dir / "teacher_generation_blocker.json", {"chunk": chunk_id, "error": "worker output did not contain JSON", "stdout_path": repo_rel(stdout_path)})
            raise RuntimeError(f"VILA worker output did not contain JSON for {chunk_id}; see {stdout_path}")
        worker_payload = json.loads(text[start:])
        write_json(output_path, worker_payload)
        results_by_path = {str(result.get("image_path")): result for result in worker_payload.get("results") or []}
        for row in chunk:
            selected = bool(row.get("selected"))
            worker_row = results_by_path.get(str(row["image_path"])) or {"error": "missing worker result"}
            teacher_comment = "" if worker_row.get("error") else worker_row_to_comment(worker_row, selected=selected)
            validation = validate_three_line(teacher_comment, selected=selected)
            merged = dict(row)
            merged.update(
                {
                    "teacher_comment_3line_raw": teacher_comment,
                    "teacher_comment_3line": teacher_comment,
                    "teacher_generation_error": worker_row.get("error"),
                    "teacher_worker_parse_mode": worker_row.get("parse_mode"),
                    "teacher_worker_short_reason": worker_row.get("short_reason"),
                    "teacher_worker_detailed_reason": worker_row.get("detailed_reason"),
                    "teacher_worker_comparison_reason": worker_row.get("comparison_reason"),
                    "teacher_worker_tags": worker_row.get("tags"),
                    "teacher_worker_confidence": worker_row.get("confidence"),
                    "teacher_worker_raw_text": worker_row.get("raw_text"),
                    "teacher_format_valid": validation["prefix_ok"] and validation["role_separation_ok"],
                    "teacher_clean_pre_audit": validation["is_clean"],
                    "teacher_source_mode": "local_vila_worker",
                    "teacher_model_name": str(model_path),
                }
            )
            append_jsonl(progress_path, merged)
            existing[str(row["teacher_subset_id"])] = merged
        write_teacher_generation_outputs(output_dir)
        print(json.dumps({"phase": "teacher_chunk_complete", "chunk": chunk_id, "completed_total": len(existing)}, ensure_ascii=False))

    write_teacher_generation_outputs(output_dir)


def write_teacher_generation_outputs(output_dir: Path) -> None:
    rows = read_jsonl(output_dir / "teacher_generation_progress.jsonl")
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "teacher_dataset_raw.csv", index=False)
    write_jsonl(output_dir / "teacher_dataset_raw.jsonl", rows)
    summary = {
        "teacher_generation_completed_count": int(len(rows)),
        "error_count": int(frame["teacher_generation_error"].notna().sum()) if "teacher_generation_error" in frame else 0,
        "clean_pre_audit_count": int(frame["teacher_clean_pre_audit"].astype(bool).sum()) if "teacher_clean_pre_audit" in frame else 0,
        "dataset_counts": {str(key): int(value) for key, value in frame["dataset_source"].value_counts().to_dict().items()} if "dataset_source" in frame else {},
        "bucket_counts": {str(key): int(value) for key, value in frame["teacher_subset_bucket"].value_counts().to_dict().items()} if "teacher_subset_bucket" in frame else {},
    }
    write_json(output_dir / "teacher_generation_summary.json", summary)


def audit_repair(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    raw_rows = read_jsonl(output_dir / "teacher_generation_progress.jsonl")
    repaired_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    counts = {
        "rows": len(raw_rows),
        "format_invalid": 0,
        "leakage": 0,
        "generic": 0,
        "role_separation_invalid": 0,
        "verdict_polarity_invalid": 0,
        "repaired_clean_rows": 0,
        "dropped_rows": 0,
    }
    for row in raw_rows:
        selected = bool(row.get("selected"))
        original = row.get("teacher_comment_3line")
        before = validate_three_line(original, selected=selected)
        generation_error = bool(row.get("teacher_generation_error"))
        counts["format_invalid"] += int(not before["prefix_ok"])
        counts["role_separation_invalid"] += int(not before["role_separation_ok"])
        counts["verdict_polarity_invalid"] += int(not before["verdict_polarity_ok"])
        counts["leakage"] += int(before["leakage"])
        counts["generic"] += int(before["generic"])
        candidate = original if before["is_clean"] else repair_comment(original, selected=selected)
        after = validate_three_line(candidate, selected=selected)
        if generation_error:
            after = dict(after, is_clean=False)
        if after["is_clean"]:
            clean = dict(row)
            clean["teacher_comment_3line_original"] = original
            clean["teacher_comment_3line"] = "\n".join(after["lines"])
            clean["teacher_comment_strength_line"] = after["lines"][0]
            clean["teacher_comment_weakness_line"] = after["lines"][1]
            clean["teacher_comment_verdict_line"] = after["lines"][2]
            clean["teacher_audit_clean"] = True
            clean["teacher_repaired"] = candidate != original
            repaired_rows.append(clean)
            counts["repaired_clean_rows"] += 1
        else:
            counts["dropped_rows"] += 1
        audit_rows.append(
            {
                "teacher_subset_id": row.get("teacher_subset_id"),
                "image_path": row.get("image_path"),
                "dataset_source": row.get("dataset_source"),
                "teacher_subset_bucket": row.get("teacher_subset_bucket"),
                "selected": selected,
                "line_count": before["line_count"],
                "prefix_ok": before["prefix_ok"],
                "role_separation_ok": before["role_separation_ok"],
                "verdict_polarity_ok": before["verdict_polarity_ok"],
                "leakage": before["leakage"],
                "generic": before["generic"],
                "teacher_generation_error": row.get("teacher_generation_error"),
                "clean_before_repair": before["is_clean"],
                "clean_after_repair": after["is_clean"],
                "teacher_comment_3line": original,
                "repaired_teacher_comment_3line": candidate,
            }
        )

    pd.DataFrame(audit_rows).to_csv(output_dir / "teacher_audit_rows.csv", index=False)
    pd.DataFrame(repaired_rows).to_csv(output_dir / "repaired_teacher_dataset.csv", index=False)
    write_jsonl(output_dir / "repaired_teacher_dataset.jsonl", repaired_rows)
    summary = {
        "summary_counts": counts,
        "clean_dataset_rows": len(repaired_rows),
        "selected_counts": {
            "selected": int(sum(1 for row in repaired_rows if bool(row.get("selected")))),
            "not_selected": int(sum(1 for row in repaired_rows if not bool(row.get("selected")))),
        },
        "bucket_counts": dict(pd.DataFrame(repaired_rows)["teacher_subset_bucket"].value_counts()) if repaired_rows else {},
        "repair_policy": "deterministic cleanup for prefix/polarity/leakage; rows still generic or leaky after repair are dropped",
    }
    summary["bucket_counts"] = {str(key): int(value) for key, value in summary["bucket_counts"].items()}
    write_json(output_dir / "teacher_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_student_messages(row: dict[str, Any], target: str) -> list[dict[str, Any]]:
    selected = "selected" if bool(row.get("selected")) else "not selected"
    band = str(row.get("teacher_subset_bucket") or row.get("selection_bucket") or "review")
    band_text = {
        "selected_high": "strong visual candidate",
        "middle_near_miss": "borderline candidate",
        "rejected_low": "weak visual candidate",
    }.get(band, "review candidate")
    system = (
        "You are the Gemma 4 E4B A-cut student. Return exactly three lines with the prefixes "
        "Strength:, Weakness:, and Verdict:. Use visible evidence only. Do not mention hidden context, "
        "scores, numbers, thresholds, rankings, percent bands, model names, bullets, or extra text."
    )
    user_text = "\n".join(
        [
            "Use the image first. The internal selection context controls only the verdict.",
            f"Internal selection state: {selected}.",
            f"Internal visual band: {band_text}.",
            "Write visible evidence for why the frame made or missed the A-cut.",
            'If selected, the Verdict line must start with "Verdict: selected because".',
            'If not selected, the Verdict line must start with "Verdict: not selected because".',
            "Return exactly 3 lines and nothing else.",
        ]
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]


def split_balanced_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    selected = [row for row in rows if bool(row.get("selected"))]
    rejected = [row for row in rows if not bool(row.get("selected"))]
    rng = np.random.default_rng(seed)
    rng.shuffle(selected)
    rng.shuffle(rejected)
    n = min(len(selected), len(rejected))
    selected = selected[:n]
    rejected = rejected[:n]
    balanced = selected + rejected
    rng.shuffle(balanced)
    train_n = int(math.floor(len(balanced) * 0.80))
    val_n = int(math.floor(len(balanced) * 0.10))
    return {
        "train": balanced[:train_n],
        "val": balanced[train_n: train_n + val_n],
        "test": balanced[train_n + val_n:],
    }


def manifest_row(row: dict[str, Any], split: str, dataset_name: str | None = None) -> dict[str, Any]:
    target = str(row["teacher_comment_3line"])
    row_id = stable_id(dataset_name or row.get("dataset_source"), split, row.get("teacher_subset_id") or Path(str(row["image_path"])).name)
    return {
        "id": row_id,
        "image": str(row["image_path"]),
        "image_path": str(row["image_path"]),
        "split": split,
        "dataset": dataset_name or str(row.get("dataset_source")),
        "selected": bool(row.get("selected")),
        "selection_bucket": row.get("teacher_subset_bucket") or row.get("selection_bucket"),
        "teacher_comment_3line": target,
        "teacher_comment_strength_line": row.get("teacher_comment_strength_line"),
        "teacher_comment_weakness_line": row.get("teacher_comment_weakness_line"),
        "teacher_comment_verdict_line": row.get("teacher_comment_verdict_line"),
        "teacher_source_mode": row.get("teacher_source_mode"),
        "teacher_model_name": row.get("teacher_model_name"),
        "messages": build_student_messages(row, target),
    }


def load_previous_test_vila_rows() -> list[dict[str, Any]]:
    path = repo_path("outputs/gemma4_e4b_aesthetic3_smoke/test_vila.jsonl")
    if not path.exists():
        return []
    rows = read_jsonl(path)
    out = []
    for row in rows:
        row = dict(row)
        row["teacher_subset_bucket"] = "test_vila_sanity"
        out.append(row)
    return out


def build_manifests(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    repaired_rows = read_jsonl(output_dir / "repaired_teacher_dataset.jsonl")
    splits = split_balanced_rows(repaired_rows, seed=int(args.seed))
    all_manifest_rows: list[dict[str, Any]] = []
    sizes: dict[str, int] = {}
    for split, split_rows in splits.items():
        payload = [manifest_row(row, split) for row in split_rows]
        write_jsonl(output_dir / f"{split}.jsonl", payload)
        pd.DataFrame(payload).drop(columns=["messages"], errors="ignore").to_csv(output_dir / f"{split}.csv", index=False)
        all_manifest_rows.extend(payload)
        sizes[split] = len(payload)

    test_vila_rows = [manifest_row(row, "test_vila", dataset_name="test_vila_sanity_previous_pass") for row in load_previous_test_vila_rows()]
    write_jsonl(output_dir / "test_vila.jsonl", test_vila_rows)
    pd.DataFrame(test_vila_rows).drop(columns=["messages"], errors="ignore").to_csv(output_dir / "test_vila.csv", index=False)
    write_jsonl(output_dir / "dataset_all.jsonl", all_manifest_rows)
    pd.DataFrame(all_manifest_rows).drop(columns=["messages"], errors="ignore").to_csv(output_dir / "dataset_manifest.csv", index=False)
    sizes["test_vila"] = len(test_vila_rows)
    sizes["dataset_all"] = len(all_manifest_rows)
    summary = {
        "manifest_sizes": sizes,
        "balanced_training_rows_used": len(all_manifest_rows),
        "clean_teacher_rows_available": len(repaired_rows),
        "selected_counts_dataset_all": {
            "selected": int(sum(1 for row in all_manifest_rows if row["selected"])),
            "not_selected": int(sum(1 for row in all_manifest_rows if not row["selected"])),
        },
        "test_vila_policy": "reused previous strict 3-line test_vila sanity manifest; excluded from train/val/test main splits",
        "student_prompt_policy": "no numeric scores, ranks, thresholds, or model names; only selection state and abstract visual band",
    }
    write_json(output_dir / "manifest_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def write_training_config(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    clean_rows = len(read_jsonl(output_dir / "repaired_teacher_dataset.jsonl"))
    dataset_all = len(read_jsonl(output_dir / "dataset_all.jsonl"))
    if clean_rows >= 300:
        epochs = 1.0
        grad_accum = 8
        save_steps = 25
        eval_steps = 25
        run_strength = "stronger"
    else:
        epochs = 1.0
        grad_accum = 8
        save_steps = 10
        eval_steps = 10
        run_strength = "short_real" if clean_rows >= 100 else "blocked_insufficient_clean_rows"
    config = {
        "experiment_name": output_dir.name,
        "dataset_dir": repo_rel(output_dir),
        "train_manifest": "train.jsonl",
        "val_manifest": "val.jsonl",
        "test_manifests": {"test": "test.jsonl", "test_vila": "test_vila.jsonl"},
        "output_dir": repo_rel(output_dir / "train_run"),
        "model_name_or_path": "google/gemma-4-E4B-it",
        "processor_name_or_path": "google/gemma-4-E4B-it",
        "base_adapter_dir": "outputs/gemma4_e4b_aesthetic3_smoke/train_run_pass2b/final_adapter",
        "allow_remote_model_download": True,
        "lora": {"enabled": True, "r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
        "quantization": {
            "load_in_4bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
        "training": {
            "num_train_epochs": epochs,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": 5e-5,
            "warmup_ratio": 0.0,
            "logging_steps": 1,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "max_grad_norm": 1.0,
            "weight_decay": 0.0,
            "seed": int(args.seed),
            "max_length": None,
        },
    }
    config_path = output_dir / "gemma_train_config.json"
    write_json(config_path, config)
    write_json(
        output_dir / "training_plan_summary.json",
        {
            "clean_teacher_rows": clean_rows,
            "dataset_all_rows": dataset_all,
            "run_strength": run_strength,
            "training_config": repo_rel(config_path),
            "can_train": clean_rows >= 100 and dataset_all >= 100,
        },
    )
    print(json.dumps({"training_config": repo_rel(config_path), "run_strength": run_strength, "can_train": clean_rows >= 100 and dataset_all >= 100}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expanded AVA/AADB Gemma 4 E4B aesthetic3 distillation pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build_pool")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ava_target", type=int, default=2000)
    p.add_argument("--aadb_target", type=int, default=1000)
    p.add_argument("--seed", type=int, default=427)
    p.set_defaults(func=build_pool)

    p = sub.add_parser("score_pool")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--nima_model", default="checkpoints/nima_ava_gpu/final_model.keras")
    p.add_argument("--rgnet_model", default="checkpoints/rgnet_aadb_gpu/final_model.keras")
    p.add_argument("--alamp_model", default="checkpoints/alamp_aadb_gpu/final_model.keras")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--rgnet_image_size", type=int, default=256)
    p.add_argument("--alamp_global_size", type=int, default=384)
    p.add_argument("--alamp_patch_size", type=int, default=224)
    p.add_argument("--alamp_num_patches", type=int, default=5)
    p.add_argument("--alamp_patch_scale", type=float, default=0.35)
    p.add_argument("--max_rows", type=int)
    p.add_argument("--progress_every", type=int, default=25)
    p.set_defaults(func=score_pool)

    p = sub.add_parser("build_review")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--section_limit", type=int, default=300)
    p.add_argument("--disagreement_limit", type=int, default=400)
    p.set_defaults(func=build_review)

    p = sub.add_parser("build_teacher_subset")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--teacher_target", type=int, default=600)
    p.add_argument("--seed", type=int, default=427)
    p.set_defaults(func=build_teacher_subset)

    p = sub.add_parser("generate_teacher")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vila_python", default=".venv_vila/bin/python")
    p.add_argument("--vila_model_path", default="checkpoints/NVILA-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--chunk_size", type=int, default=10)
    p.add_argument("--max_rows", type=int)
    p.set_defaults(func=generate_teacher)

    p = sub.add_parser("audit_repair")
    p.add_argument("--output_dir", required=True)
    p.set_defaults(func=audit_repair)

    p = sub.add_parser("build_manifests")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=427)
    p.set_defaults(func=build_manifests)

    p = sub.add_parser("write_training_config")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=427)
    p.set_defaults(func=write_training_config)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
