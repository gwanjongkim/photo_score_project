# TechIQA-Guard v1 단일 출력 학습 매니페스트를 생성하는 스크립트
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

MANUAL_FALSE_POSITIVES = [
    "20230201_181300.jpg",
    "1675342165226-13.jpg",
    "1675342165226-3.jpg",
]

OUTPUT_COLUMNS = [
    "image_path",
    "filename",
    "relative_path",
    "dataset",
    "normalized_mos",
    "mos_100",
    "split",
    "hard_false_positive",
    "strong_hard_false_positive",
    "manual_false_positive",
    "missing_image",
    "existing_combined_score",
    "koniq_teacher_score",
    "flive_teacher_score",
    "mixed112_score",
    "delta_mixed112_existing",
    "guard_source_score_100",
    "guard_cap_100",
    "false_positive_margin",
    "source_note",
]

DEFAULT_PATHS = {
    "spaq_all": "data/processed/spaq/labels_all.csv",
    "koniq_all": "data/processed/koniq10k/labels_all.csv",
    "flive_all": "data/processed/flive/labels_image_all.csv",
    "mixed_train": "data/processed/topiq_replacement/mixed_112_train.csv",
    "mixed_val": "data/processed/topiq_replacement/mixed_112_val.csv",
    "flive_test": "data/processed/flive/image_test.csv",
    "koniq_test": "data/processed/koniq10k/splits/test.csv",
    "spaq_test": "data/processed/spaq/test.csv",
}

PREDICTION_DIR = REPO_ROOT / "outputs/eval_final_topiq_candidates_vs_existing_technical_20260520"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TechIQA-Guard v1 single-output dataset manifests."
    )
    parser.add_argument("--out_dir", default="data/processed/techiqa_guard")
    parser.add_argument("--spaq_all")
    parser.add_argument("--koniq_all")
    parser.add_argument("--flive_all")
    parser.add_argument("--mixed_train")
    parser.add_argument("--mixed_val")
    parser.add_argument("--flive_test")
    parser.add_argument("--koniq_test")
    parser.add_argument("--spaq_test")
    parser.add_argument("--comparison_log")
    parser.add_argument("--comparison_report")
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_size", type=int, default=450)
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_existing_path(value: str | None, default_key: str, warnings: list[str]) -> Path | None:
    raw = value or DEFAULT_PATHS[default_key]
    path = repo_path(raw)
    if path.exists():
        return path
    warnings.append(f"Missing input for {default_key}: {display_path(path)}")
    return None


def read_csv(path: Path, label: str, input_files_used: dict[str, str | list[str]]) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[columns] {label}: {display_path(path)} -> {list(df.columns)}")
    input_files_used[label] = display_path(path)
    return df


def normalize_dataset_name(value: object, hint: str | None = None) -> str:
    raw = str(value if value is not None and not pd.isna(value) else hint or "").lower()
    if "hard" in raw:
        return "hard_fp"
    if "koniq" in raw:
        return "koniq"
    if "flive" in raw:
        return "flive"
    if "spaq" in raw:
        return "spaq"
    return str(hint or raw or "unknown")


def resolve_image_path(raw: object, csv_path: Path | None = None) -> str:
    if raw is None or pd.isna(raw):
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    p = Path(text)
    if p.is_absolute():
        return str(p)
    candidates = []
    if csv_path is not None:
        candidates.append((csv_path.parent / p).resolve())
    candidates.append((REPO_ROOT / p).resolve())
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1])


def relative_image_path(image_path: str, fallback: object = "") -> str:
    if fallback is not None and not pd.isna(fallback) and str(fallback).strip():
        return str(fallback)
    if not image_path:
        return ""
    p = Path(image_path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return p.name


def first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def numeric_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def normalize_scores(df: pd.DataFrame, warnings: list[str], label: str) -> tuple[pd.Series, pd.Series]:
    norm_col = first_existing_column(df, ["normalized_mos", "normalized_score"])
    mos100_col = first_existing_column(df, ["mos_100", "score_100"])
    mos_col = first_existing_column(df, ["mos", "mean_score", "y_true"])

    normalized = numeric_series(df, norm_col)
    mos_100 = numeric_series(df, mos100_col)
    mos_raw = numeric_series(df, mos_col)

    has_norm = normalized.notna()
    has_mos100 = mos_100.notna()
    has_mos = mos_raw.notna()

    mos_raw_looks_unit = bool(has_mos.any() and mos_raw.dropna().max() <= 1.01)

    normalized = normalized.where(has_norm)
    normalized = normalized.where(has_norm, mos_100 / 100.0)
    normalized = normalized.where(has_norm | has_mos100, np.where(mos_raw_looks_unit, mos_raw, mos_raw / 100.0))

    mos_100 = mos_100.where(has_mos100)
    mos_100 = mos_100.where(has_mos100, normalized * 100.0)
    mos_100 = mos_100.where(has_mos100 | has_norm, np.where(mos_raw_looks_unit, mos_raw * 100.0, mos_raw))

    bad_norm = normalized.notna() & ((normalized < 0.0) | (normalized > 1.0))
    bad_mos = mos_100.notna() & ((mos_100 < 0.0) | (mos_100 > 100.0))
    if bad_norm.any() or bad_mos.any():
        warnings.append(
            f"{label}: clipped {int(bad_norm.sum())} normalized_mos and {int(bad_mos.sum())} mos_100 values into range"
        )
    return normalized.clip(0.0, 1.0), mos_100.clip(0.0, 100.0)


def standardize_manifest(
    df: pd.DataFrame,
    csv_path: Path,
    dataset_hint: str,
    split: str,
    allow_missing: bool,
    warnings: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    image_col = first_existing_column(df, ["image_path", "path", "filepath", "file_path"])
    relative_col = first_existing_column(df, ["relative_path", "rel_path"])
    dataset_col = first_existing_column(df, ["dataset", "source_dataset"])

    if image_col is None:
        warnings.append(f"{display_path(csv_path)} has no image path column; using filename-only missing rows")
        out["image_path"] = ""
    else:
        out["image_path"] = df[image_col].map(lambda p: resolve_image_path(p, csv_path))

    out["filename"] = out["image_path"].map(lambda p: Path(p).name if p else "")
    if relative_col is not None:
        out["relative_path"] = [
            relative_image_path(image_path, fallback)
            for image_path, fallback in zip(out["image_path"], df[relative_col], strict=False)
        ]
    else:
        out["relative_path"] = out["image_path"].map(relative_image_path)

    if dataset_col is None:
        out["dataset"] = normalize_dataset_name(None, dataset_hint)
    else:
        out["dataset"] = df[dataset_col].map(lambda v: normalize_dataset_name(v, dataset_hint))

    out["normalized_mos"], out["mos_100"] = normalize_scores(df, warnings, display_path(csv_path))
    out["split"] = split
    out["hard_false_positive"] = False
    out["strong_hard_false_positive"] = False
    out["manual_false_positive"] = False
    out["missing_image"] = out["image_path"].map(lambda p: not bool(p) or not Path(p).exists())
    out["existing_combined_score"] = np.nan
    out["koniq_teacher_score"] = np.nan
    out["flive_teacher_score"] = np.nan
    out["mixed112_score"] = np.nan
    out["delta_mixed112_existing"] = np.nan
    out["guard_source_score_100"] = np.nan
    out["guard_cap_100"] = np.nan
    out["false_positive_margin"] = np.nan
    out["source_note"] = f"manifest:{display_path(csv_path)}"

    missing_count = int(out["missing_image"].sum())
    if missing_count and not allow_missing:
        warnings.append(f"{display_path(csv_path)}: dropped {missing_count} rows with missing image paths")
        out = out.loc[~out["missing_image"]].copy()
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def load_manifest_or_empty(
    path: Path | None,
    label: str,
    dataset_hint: str,
    split: str,
    allow_missing: bool,
    warnings: list[str],
    input_files_used: dict[str, str | list[str]],
) -> pd.DataFrame:
    if path is None:
        warnings.append(f"No {label} manifest was available; writing empty {split} output for that source")
        return empty_output()
    df = read_csv(path, label, input_files_used)
    return standardize_manifest(df, path, dataset_hint, split, allow_missing, warnings)


def prediction_file(dataset: str, model: str) -> Path:
    return PREDICTION_DIR / f"predictions_{model}_{dataset}.csv"


def prediction_score_frame(path: Path, score_col: str, label: str, input_files_used: dict[str, str | list[str]]) -> pd.DataFrame:
    df = read_csv(path, label, input_files_used)
    pred_col = first_existing_column(df, ["y_pred", "pred", "score", "score_100"])
    true_col = first_existing_column(df, ["y_true", "mos", "mos_100"])
    image_col = first_existing_column(df, ["image_path", "path", "filepath", "file_path"])
    if pred_col is None or image_col is None:
        return pd.DataFrame(columns=["image_path", score_col, "prediction_true_100"])

    out = pd.DataFrame()
    out["image_path"] = df[image_col].map(lambda p: resolve_image_path(p, path))
    out[score_col] = pd.to_numeric(df[pred_col], errors="coerce")
    if true_col is not None:
        true_score = pd.to_numeric(df[true_col], errors="coerce")
        if true_score.dropna().max() <= 1.01:
            true_score = true_score * 100.0
        out["prediction_true_100"] = true_score
    return out


def load_prediction_bundle(input_files_used: dict[str, str | list[str]], warnings: list[str]) -> dict[str, pd.DataFrame]:
    if not PREDICTION_DIR.exists():
        warnings.append(f"Prediction directory not found: {display_path(PREDICTION_DIR)}")
        return {}

    bundle: dict[str, pd.DataFrame] = {}
    specs = {
        "existing_combined_score": "existing_avg_technical",
        "koniq_teacher_score": "koniq_mobile",
        "flive_teacher_score": "flive_image_mobile",
        "mixed112_score": "topiq_lite_mixed112_frozen_fp16",
    }
    for dataset in ["flive", "koniq", "spaq"]:
        merged: pd.DataFrame | None = None
        for score_col, model in specs.items():
            path = prediction_file(dataset, model)
            if not path.exists():
                warnings.append(f"Missing prediction file: {display_path(path)}")
                continue
            frame = prediction_score_frame(
                path,
                score_col,
                f"prediction_{dataset}_{model}",
                input_files_used,
            )
            if merged is None:
                merged = frame
            else:
                merged = merged.merge(frame, on="image_path", how="outer", suffixes=("", "_dup"))
                if "prediction_true_100_dup" in merged.columns:
                    merged["prediction_true_100"] = merged["prediction_true_100"].where(
                        merged["prediction_true_100"].notna(), merged["prediction_true_100_dup"]
                    )
                    merged = merged.drop(columns=["prediction_true_100_dup"])
        if merged is None:
            continue
        merged["dataset"] = dataset
        merged["delta_mixed112_existing"] = merged["mixed112_score"] - merged["existing_combined_score"]
        bundle[dataset] = merged
    return bundle


def attach_predictions(manifest: pd.DataFrame, prediction_df: pd.DataFrame | None) -> pd.DataFrame:
    if prediction_df is None or prediction_df.empty or manifest.empty:
        return manifest
    score_cols = [
        "existing_combined_score",
        "koniq_teacher_score",
        "flive_teacher_score",
        "mixed112_score",
        "delta_mixed112_existing",
    ]
    attach = prediction_df[["image_path", *score_cols]].drop_duplicates("image_path")
    merged = manifest.drop(columns=score_cols).merge(attach, on="image_path", how="left")
    hard = merged["delta_mixed112_existing"] >= 5.0
    strong = merged["delta_mixed112_existing"] >= 8.0
    merged["hard_false_positive"] = hard.fillna(False)
    merged["strong_hard_false_positive"] = strong.fillna(False)
    guard_source = merged["existing_combined_score"].where(
        merged["existing_combined_score"].notna(), merged["flive_teacher_score"]
    )
    merged["guard_source_score_100"] = np.where(hard, guard_source, np.nan)
    merged["false_positive_margin"] = np.where(hard, 5.0, np.nan)
    merged["guard_cap_100"] = np.where(hard, guard_source + 5.0, np.nan)
    return merged[OUTPUT_COLUMNS]


def image_lookup_from_manifests(frames: Iterable[pd.DataFrame]) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for frame in frames:
        for row in frame.to_dict("records"):
            keys = [row.get("image_path", ""), row.get("filename", "")]
            for key in keys:
                if key and key not in lookup:
                    lookup[str(key)] = row
    return lookup


def discover_manual_image(filename: str) -> tuple[str, list[str]]:
    matches: list[Path] = []
    for root_name in ["data", "outputs"]:
        root = REPO_ROOT / root_name
        if root.exists():
            matches.extend(root.rglob(filename))
    matches = sorted({p.resolve() for p in matches})
    if not matches:
        return "", []
    return str(matches[0]), [display_path(p) for p in matches]


def make_manual_rows(
    lookup: dict[str, dict[str, object]],
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    status: dict[str, dict[str, object]] = {}
    for filename in MANUAL_FALSE_POSITIVES:
        image_path, matches = discover_manual_image(filename)
        source = lookup.get(image_path) or lookup.get(filename) or {}
        missing = not bool(image_path) or not Path(image_path).exists()
        if missing:
            warnings.append(f"Manual false-positive image not found under data/ or outputs/: {filename}")

        normalized_mos = source.get("normalized_mos", np.nan)
        mos_100 = source.get("mos_100", np.nan)
        if pd.isna(normalized_mos) and not pd.isna(mos_100):
            normalized_mos = float(mos_100) / 100.0
        if pd.isna(mos_100) and not pd.isna(normalized_mos):
            mos_100 = float(normalized_mos) * 100.0

        row = {
            "image_path": image_path,
            "filename": filename,
            "relative_path": relative_image_path(image_path, filename),
            "dataset": "hard_fp",
            "normalized_mos": normalized_mos,
            "mos_100": mos_100,
            "split": "hard_fp",
            "hard_false_positive": True,
            "strong_hard_false_positive": False,
            "manual_false_positive": True,
            "missing_image": missing,
            "existing_combined_score": np.nan,
            "koniq_teacher_score": np.nan,
            "flive_teacher_score": np.nan,
            "mixed112_score": np.nan,
            "delta_mixed112_existing": np.nan,
            "guard_source_score_100": np.nan,
            "guard_cap_100": np.nan,
            "false_positive_margin": 5.0,
            "source_note": "manual_user_flagged",
        }
        rows.append(row)
        status[filename] = {"found": not missing, "selected_path": display_path(Path(image_path)) if image_path else None, "all_matches": matches}
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), status


def prediction_delta_counts(prediction_bundle: dict[str, pd.DataFrame]) -> dict[str, int]:
    hard_count = 0
    strong_count = 0
    for frame in prediction_bundle.values():
        hard_count += int((frame["delta_mixed112_existing"] >= 5.0).sum())
        strong_count += int((frame["delta_mixed112_existing"] >= 8.0).sum())
    return {"hard": hard_count, "strong": strong_count}


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def discover_comparison_files(exclude_roots: Iterable[Path] = ()) -> list[Path]:
    files: set[Path] = set()
    names = {
        "topiq_mixed112_compare_report.md",
        "topiq_mixed112_runtime_compare.log",
        "topiq_mixed112_profile_compare.log",
    }
    for root_name in ["data", "outputs"]:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for name in names:
            for path in root.rglob(name):
                if not any(path_is_under(path, excluded) for excluded in exclude_roots):
                    files.add(path)
        for suffix in ["*.log", "*.md", "*.txt"]:
            for path in root.rglob(suffix):
                if any(path_is_under(path, excluded) for excluded in exclude_roots):
                    continue
                try:
                    if path.stat().st_size > 20_000_000:
                        continue
                    if "[TechnicalIqaCompare]" in path.read_text(errors="ignore"):
                        files.add(path)
                except OSError:
                    continue
    return sorted(files)


def score_from_line(patterns: Iterable[str], line: str) -> float:
    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value * 100.0 if value <= 1.01 else value
    return math.nan


def parse_comparison_text_rows(paths: Iterable[Path], input_files_used: dict[str, str | list[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    used: list[str] = []
    image_re = re.compile(r"([A-Za-z0-9_().-]+\.(?:jpg|jpeg|png|webp))", flags=re.IGNORECASE)
    for path in paths:
        used.append(display_path(path))
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        print(f"[comparison] scanning {display_path(path)}")
        for line in lines:
            if "[TechnicalIqaCompare]" not in line and "mixed112" not in line.lower():
                continue
            image_match = image_re.search(line)
            if not image_match:
                continue
            filename = image_match.group(1)
            mixed = score_from_line([r"mixed112[^0-9]*([0-9]+(?:\.[0-9]+)?)", r"mixed_112[^0-9]*([0-9]+(?:\.[0-9]+)?)"], line)
            existing = score_from_line(
                [
                    r"existing(?:_combined|_avg| avg| combined)?[^0-9]*([0-9]+(?:\.[0-9]+)?)",
                    r"combined[^0-9]*([0-9]+(?:\.[0-9]+)?)",
                ],
                line,
            )
            koniq = score_from_line([r"koniq(?:_teacher|_mobile)?[^0-9]*([0-9]+(?:\.[0-9]+)?)"], line)
            flive = score_from_line([r"flive(?:_teacher|_image|_mobile)?[^0-9]*([0-9]+(?:\.[0-9]+)?)"], line)
            if math.isnan(existing) and not math.isnan(flive):
                existing = flive
            delta = mixed - existing if not math.isnan(mixed) and not math.isnan(existing) else math.nan
            if math.isnan(delta) or delta < 5.0:
                continue
            image_path, _ = discover_manual_image(filename)
            guard_source = existing if not math.isnan(existing) else flive
            rows.append(
                {
                    "image_path": image_path,
                    "filename": filename,
                    "relative_path": relative_image_path(image_path, filename),
                    "dataset": "hard_fp",
                    "normalized_mos": np.nan,
                    "mos_100": np.nan,
                    "split": "hard_fp",
                    "hard_false_positive": True,
                    "strong_hard_false_positive": bool(delta >= 8.0),
                    "manual_false_positive": False,
                    "missing_image": not bool(image_path) or not Path(image_path).exists(),
                    "existing_combined_score": existing,
                    "koniq_teacher_score": koniq,
                    "flive_teacher_score": flive,
                    "mixed112_score": mixed,
                    "delta_mixed112_existing": delta,
                    "guard_source_score_100": guard_source,
                    "guard_cap_100": guard_source + 5.0 if not math.isnan(guard_source) else np.nan,
                    "false_positive_margin": 5.0,
                    "source_note": f"mined_comparison_text:{display_path(path)}",
                }
            )
    if used:
        input_files_used["comparison_text"] = used
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def merge_hard_fp_rows(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    rows = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    if rows.empty:
        return empty_output()
    rows["_dedupe_key"] = np.where(rows["image_path"].astype(str) != "", rows["image_path"], rows["filename"])
    rows["_manual_rank"] = rows["manual_false_positive"].astype(int)
    rows = rows.sort_values(["_dedupe_key", "_manual_rank"], ascending=[True, False])
    rows = rows.drop_duplicates("_dedupe_key", keep="first").drop(columns=["_dedupe_key", "_manual_rank"])
    rows["guard_source_score_100"] = rows["existing_combined_score"].where(
        rows["existing_combined_score"].notna(), rows["flive_teacher_score"]
    )
    rows["false_positive_margin"] = 5.0
    rows["guard_cap_100"] = rows["guard_source_score_100"] + rows["false_positive_margin"]
    rows["hard_false_positive"] = True
    rows["dataset"] = "hard_fp"
    return rows[OUTPUT_COLUMNS].reset_index(drop=True)


def sample_quality_bins(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.head(0)
    available = frame.drop_duplicates("image_path").copy()
    if len(available) <= n:
        return available
    bins = [
        available[available["normalized_mos"] < 0.333],
        available[(available["normalized_mos"] >= 0.333) & (available["normalized_mos"] < 0.667)],
        available[available["normalized_mos"] >= 0.667],
    ]
    parts: list[pd.DataFrame] = []
    base = n // 3
    remainder = n - base * 3
    for idx, bin_df in enumerate(bins):
        want = base + (1 if idx < remainder else 0)
        if not bin_df.empty and want:
            parts.append(bin_df.sample(n=min(want, len(bin_df)), random_state=seed + idx))
    sampled = pd.concat(parts, ignore_index=True) if parts else available.head(0)
    if len(sampled) < n:
        remaining = available.loc[~available["image_path"].isin(sampled["image_path"])]
        fill = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=seed + 99)
        sampled = pd.concat([sampled, fill], ignore_index=True)
    return sampled.head(n)


def build_smoke(train: pd.DataFrame, val: pd.DataFrame, hard_fp: pd.DataFrame, smoke_size: int, seed: int) -> pd.DataFrame:
    base = pd.concat([train, val], ignore_index=True)
    base = base.loc[base["dataset"].isin(["spaq", "koniq", "flive"])].copy()
    hard = hard_fp.copy()
    remaining = max(smoke_size - len(hard), 0)
    datasets = ["spaq", "koniq", "flive"]
    per_dataset = {name: remaining // len(datasets) for name in datasets}
    for name in datasets[: remaining % len(datasets)]:
        per_dataset[name] += 1

    sampled = [
        sample_quality_bins(base.loc[base["dataset"] == dataset], per_dataset[dataset], seed + idx * 17)
        for idx, dataset in enumerate(datasets)
    ]
    smoke = pd.concat([*sampled, hard], ignore_index=True)
    smoke = smoke.drop_duplicates(["image_path", "filename"], keep="first")
    return smoke.sample(frac=1.0, random_state=seed).reset_index(drop=True)[OUTPUT_COLUMNS]


def bool_counts(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].fillna(False).astype(bool).sum()) if column in frame.columns else 0


def write_readme(out_dir: Path, summary: dict[str, object]) -> None:
    lines = [
        "# TechIQA-Guard v1 Dataset",
        "",
        "Generated manifests for direct single-output technical IQA training.",
        "",
        "This is dataset preparation only. It is not teacher-student distillation, multi-head training, model training, or TFLite export.",
        "",
        "## Hard False-Positive Rules",
        "",
        "- `hard_false_positive`: `delta_mixed112_existing >= 5` or manual user flag.",
        "- `strong_hard_false_positive`: `delta_mixed112_existing >= 8`.",
        "- Manual user-flagged files are always included even when MOS or images are missing.",
        "- Original MOS is preserved when available. Hard-FP guard columns carry cap metadata.",
        "",
        "## Files",
        "",
        "- `train.csv` and `val.csv`: direct single-head training manifests from mixed_112 inputs when available.",
        "- `test_flive.csv`, `test_koniq.csv`, `test_spaq.csv`: separate held-out test manifests.",
        "- `hard_false_positive.csv`: manual and mined hard false-positive rows.",
        "- `smoke_v1.csv`: balanced smoke manifest plus all hard false positives.",
        "- `summary.json`: counts, inputs, manual file discovery status, and warnings.",
        "",
        "## Row Counts",
        "",
    ]
    row_counts = summary.get("row_counts", {})
    if isinstance(row_counts, dict):
        for name, count in row_counts.items():
            lines.append(f"- `{name}`: {count}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    input_files_used: dict[str, str | list[str]] = {}

    paths = {
        key: resolve_existing_path(getattr(args, key), key, warnings)
        for key in DEFAULT_PATHS
    }

    spaq_all = load_manifest_or_empty(paths["spaq_all"], "spaq_all", "spaq", "all", True, warnings, input_files_used)
    koniq_all = load_manifest_or_empty(paths["koniq_all"], "koniq_all", "koniq", "all", True, warnings, input_files_used)
    flive_all = load_manifest_or_empty(paths["flive_all"], "flive_all", "flive", "all", True, warnings, input_files_used)

    train = load_manifest_or_empty(
        paths["mixed_train"], "mixed_train", "mixed", "train", args.allow_missing, warnings, input_files_used
    )
    val = load_manifest_or_empty(paths["mixed_val"], "mixed_val", "mixed", "val", args.allow_missing, warnings, input_files_used)
    test_flive = load_manifest_or_empty(
        paths["flive_test"], "flive_test", "flive", "test", args.allow_missing, warnings, input_files_used
    )
    test_koniq = load_manifest_or_empty(
        paths["koniq_test"], "koniq_test", "koniq", "test", args.allow_missing, warnings, input_files_used
    )
    test_spaq = load_manifest_or_empty(
        paths["spaq_test"], "spaq_test", "spaq", "test", args.allow_missing, warnings, input_files_used
    )

    prediction_bundle = load_prediction_bundle(input_files_used, warnings)
    test_flive = attach_predictions(test_flive, prediction_bundle.get("flive"))
    test_koniq = attach_predictions(test_koniq, prediction_bundle.get("koniq"))
    test_spaq = attach_predictions(test_spaq, prediction_bundle.get("spaq"))

    lookup = image_lookup_from_manifests([spaq_all, koniq_all, flive_all, train, val, test_flive, test_koniq, test_spaq])
    manual_rows, manual_status = make_manual_rows(lookup, warnings)
    eval_delta_counts = prediction_delta_counts(prediction_bundle)

    comparison_paths: list[Path] = []
    if args.comparison_log:
        comparison_paths.append(repo_path(args.comparison_log))
    if args.comparison_report:
        comparison_paths.append(repo_path(args.comparison_report))
    if not comparison_paths:
        comparison_paths = discover_comparison_files(exclude_roots=[out_dir])
        if not comparison_paths:
            warnings.append("No topiq_mixed112 comparison report/log or [TechnicalIqaCompare] file was found under data/ or outputs/")
    comparison_paths = [p for p in comparison_paths if p.exists()]
    comparison_rows = parse_comparison_text_rows(comparison_paths, input_files_used)

    hard_fp = merge_hard_fp_rows([comparison_rows, manual_rows])
    smoke = build_smoke(train, val, hard_fp, args.smoke_size, args.seed)

    outputs = {
        "train": train,
        "val": val,
        "test_flive": test_flive,
        "test_koniq": test_koniq,
        "test_spaq": test_spaq,
        "hard_false_positive": hard_fp,
        "smoke_v1": smoke,
    }

    for name, frame in outputs.items():
        write_csv(out_dir / f"{name}.csv", frame[OUTPUT_COLUMNS])

    primary_outputs = {
        name: frame
        for name, frame in outputs.items()
        if name in {"train", "val", "test_flive", "test_koniq", "test_spaq", "hard_false_positive"}
    }
    combined = pd.concat(primary_outputs.values(), ignore_index=True)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": display_path(out_dir),
        "row_counts": {name: int(len(frame)) for name, frame in outputs.items()},
        "counts_by_dataset": {str(k): int(v) for k, v in combined["dataset"].value_counts(dropna=False).items()},
        "counts_by_split": {str(k): int(v) for k, v in combined["split"].value_counts(dropna=False).items()},
        "hard_fp_count": bool_counts(hard_fp, "hard_false_positive"),
        "strong_hard_fp_count": bool_counts(hard_fp, "strong_hard_false_positive"),
        "manual_fp_count": bool_counts(hard_fp, "manual_false_positive"),
        "eval_prediction_delta_candidate_count": eval_delta_counts["hard"],
        "eval_prediction_strong_delta_candidate_count": eval_delta_counts["strong"],
        "missing_image_count": bool_counts(combined, "missing_image"),
        "manual_false_positive_files": manual_status,
        "input_files_used": input_files_used,
        "warnings": warnings,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_readme(out_dir, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
