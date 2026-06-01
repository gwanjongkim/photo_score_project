# A-cut ICAA 제거 후 미학 앙상블 가중치를 벤치마크하는 스크립트
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR_DEFAULT = REPO_ROOT / "outputs/acut_icaa_skip_weight_search_20260601"
SOURCE_SCORES_DEFAULT = REPO_ROOT / "outputs/test_vila_aesthetic_direct_compare_20260504_135316/old_three_model_scores.csv"
ICAA_SAVED_MODEL_DEFAULT = REPO_ROOT / "outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model"

REFERENCE_WEIGHTS = {
    "nima": 0.10,
    "rgnet": 0.40,
    "alamp": 0.30,
    "icaa": 0.20,
}

SPECIAL_CANDIDATES = {
    "proposed_020_030_050": (0.20, 0.30, 0.50),
    "old_no_icaa_renormalized": (0.125, 0.50, 0.375),
    "equal": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
}


@dataclass(frozen=True)
class MetricRow:
    candidate_name: str
    nima_weight: float
    rgnet_weight: float
    alamp_weight: float
    subset: str
    image_count: int
    mae: float
    maxae: float
    rmse: float
    spearman: float
    pearson: float
    top1_match: bool
    top3_overlap: int
    top3_overlap_fraction: float
    top5_overlap: int
    top5_overlap_fraction: float
    top10_overlap: int
    top10_overlap_fraction: float
    ranking_inversions: int
    nan_count: int
    range_error_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_scores_csv", type=Path, default=SOURCE_SCORES_DEFAULT)
    parser.add_argument("--icaa_saved_model", type=Path, default=ICAA_SAVED_MODEL_DEFAULT)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grid_step", type=float, default=0.05)
    parser.add_argument("--refine_step", type=float, default=0.01)
    parser.add_argument("--refine_radius", type=float, default=0.05)
    parser.add_argument("--reference_icaa_output", choices=["color", "mos"], default="color")
    return parser.parse_args()


def normalize_unit_score(value: float) -> float:
    if pd.isna(value):
        return np.nan
    value = float(value)
    if -0.05 <= value <= 1.05:
        return float(np.clip(value, 0.0, 1.0))
    if 1.0 < value <= 10.5:
        return float(np.clip((value - 1.0) / 9.0, 0.0, 1.0))
    if 10.5 < value <= 100.0:
        return float(np.clip(value / 100.0, 0.0, 1.0))
    return float(np.clip(value, 0.0, 1.0))


def resolve_path(path_value: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_artifact_inventory(out_dir: Path, source_scores_csv: Path, icaa_saved_model: Path) -> pd.DataFrame:
    candidates = [
        ("score_cache", source_scores_csv, "selected 50-image NIMA/RGNet/A-LAMP cache"),
        ("score_cache", REPO_ROOT / "outputs/test_vila_aesthetic_ensemble_experiment_20260423/raw_bundle_scores.csv", "alternate 50-image cache"),
        ("score_cache", REPO_ROOT / "outputs/test_vila_raw_aesthetic_scores_html_20260424/raw_aesthetic_scores.csv", "alternate 50-image aesthetic cache"),
        ("score_cache", REPO_ROOT / "outputs/acut_stage4_nima_musiq_alamp_rgnet/scores.csv", "15-image A-cut cache without ICAA"),
        ("score_cache", REPO_ROOT / "outputs/acut_stage5_full_with_pairwise/scores.csv", "15-image A-cut cache without ICAA"),
        ("model", icaa_saved_model, "TF-native ICAA SavedModel used for WSL scoring"),
        ("model", REPO_ROOT / "forWeights2/models/aesthetic/icaa_dat_tf_native_fp16.tflite", "reported production FP16 ICAA TFLite"),
        ("config", REPO_ROOT / "configs/aesthetic_weight_lab.yaml", "local aesthetic weight lab config"),
        ("config", REPO_ROOT / "configs/aesthetic_weight_lab_alamp_finetuned.yaml", "local aesthetic weight lab config"),
        ("code_readonly", REPO_ROOT / "src/infer/select_best_shots.py", "server-side selector weights inspected read-only"),
        ("code_readonly", Path("/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/lib/feature/a_cut/layer/evaluation/aesthetic_ensemble_scoring_service.dart"), "accessible Flutter aesthetic scorer inspected read-only"),
        ("code_readonly", Path("/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/lib/feature/a_cut/layer/inference/aesthetic_model_contract.dart"), "accessible Flutter model contracts inspected read-only"),
    ]
    rows = []
    for artifact_type, path, note in candidates:
        exists = path.exists()
        row = {
            "artifact_type": artifact_type,
            "path": str(path),
            "exists": exists,
            "rows": "",
            "columns": "",
            "notes": note,
        }
        if exists and path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path, nrows=5)
                row["rows"] = int(sum(1 for _ in open(path, "rb"))) - 1
                row["columns"] = "|".join(df.columns.astype(str).tolist())
            except Exception as exc:  # noqa: BLE001
                row["notes"] = f"{note}; csv inspection failed: {exc}"
        rows.append(row)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(out_dir / "artifact_inventory.csv", index=False)
    return inventory


def load_base_scores(source_scores_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(source_scores_csv)
    required = {"image_path", "nima_mean_score", "rgnet_score", "alamp_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"source score cache missing required columns: {missing}")

    rows = []
    for idx, row in df.iterrows():
        image_path = resolve_path(row["image_path"])
        rows.append(
            {
                "image_id": f"{idx:04d}_{image_path.stem}",
                "image_path": str(image_path),
                "source_image_path": row["image_path"],
                "nima_raw_score": row["nima_mean_score"],
                "nima_score": normalize_unit_score(row["nima_mean_score"]),
                "rgnet_score": normalize_unit_score(row["rgnet_score"]),
                "alamp_score": normalize_unit_score(row["alamp_score"]),
                "image_exists": image_path.exists(),
            }
        )
    score_df = pd.DataFrame(rows)
    missing_images = score_df.loc[~score_df["image_exists"], "image_path"].tolist()
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(f"{len(missing_images)} image paths from score cache do not exist; first paths: {preview}")
    return score_df


def filter_readable_images(score_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    readable_rows = []
    excluded_rows = []
    for _, row in score_df.iterrows():
        try:
            with Image.open(row["image_path"]) as image:
                image.verify()
            readable_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            excluded = row.to_dict()
            excluded["exclude_reason"] = f"unreadable image: {exc}"
            excluded_rows.append(excluded)
    if excluded_rows:
        pd.DataFrame(excluded_rows).to_csv(out_dir / "excluded_images.csv", index=False)
    else:
        pd.DataFrame(columns=list(score_df.columns) + ["exclude_reason"]).to_csv(out_dir / "excluded_images.csv", index=False)
    if not readable_rows:
        raise ValueError("all images from the selected score cache were unreadable")
    return pd.DataFrame(readable_rows).reset_index(drop=True)


def preprocess_icaa_image(image_path: str) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = Image.open(image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - mean) / std


def run_icaa_saved_model(score_df: pd.DataFrame, saved_model_path: Path, batch_size: int) -> pd.DataFrame:
    import tensorflow as tf

    if not saved_model_path.exists():
        raise FileNotFoundError(f"ICAA SavedModel does not exist: {saved_model_path}")

    loaded = tf.saved_model.load(str(saved_model_path))
    infer = loaded.signatures["serving_default"]
    mos_scores: list[float] = []
    color_scores: list[float] = []

    paths = score_df["image_path"].tolist()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = np.stack([preprocess_icaa_image(path) for path in batch_paths], axis=0).astype(np.float32)
        outputs = infer(image=tf.convert_to_tensor(batch))
        scores = outputs["scores"].numpy()
        if scores.ndim != 2 or scores.shape[1] < 2:
            raise ValueError(f"unexpected ICAA output shape: {scores.shape}")
        mos_scores.extend(scores[:, 0].astype(float).tolist())
        color_scores.extend(scores[:, 1].astype(float).tolist())

    result = score_df.copy()
    result["icaa_mos_score_raw"] = mos_scores
    result["icaa_color_score_raw"] = color_scores
    result["icaa_mos_score"] = np.clip(result["icaa_mos_score_raw"].astype(float), 0.0, 1.0)
    result["icaa_color_score"] = np.clip(result["icaa_color_score_raw"].astype(float), 0.0, 1.0)
    return result


def add_reference_and_special_scores(score_df: pd.DataFrame, reference_icaa_output: str) -> pd.DataFrame:
    df = score_df.copy()
    df["icaa_score"] = df["icaa_color_score"] if reference_icaa_output == "color" else df["icaa_mos_score"]
    df["full_reference_aesthetic_score"] = (
        REFERENCE_WEIGHTS["nima"] * df["nima_score"]
        + REFERENCE_WEIGHTS["rgnet"] * df["rgnet_score"]
        + REFERENCE_WEIGHTS["alamp"] * df["alamp_score"]
        + REFERENCE_WEIGHTS["icaa"] * df["icaa_score"]
    )
    for name, weights in SPECIAL_CANDIDATES.items():
        df[f"candidate_{name}_score"] = candidate_scores(df, weights)
    return df


def candidate_scores(df: pd.DataFrame, weights: tuple[float, float, float]) -> np.ndarray:
    n, r, a = weights
    return (n * df["nima_score"].to_numpy(float)) + (r * df["rgnet_score"].to_numpy(float)) + (a * df["alamp_score"].to_numpy(float))


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return correlation(rankdata_average(x), rankdata_average(y))


def top_overlap(reference: np.ndarray, candidate: np.ndarray, k: int) -> tuple[int, float]:
    k = min(k, len(reference))
    ref_top = set(np.argsort(-reference, kind="mergesort")[:k].tolist())
    cand_top = set(np.argsort(-candidate, kind="mergesort")[:k].tolist())
    overlap = len(ref_top & cand_top)
    return overlap, float(overlap / k) if k else np.nan


def inversion_count(reference: np.ndarray, candidate: np.ndarray) -> int:
    count = 0
    n = len(reference)
    for i in range(n):
        ref_i = reference[i]
        cand_i = candidate[i]
        for j in range(i + 1, n):
            ref_delta = ref_i - reference[j]
            cand_delta = cand_i - candidate[j]
            if ref_delta == 0.0 or cand_delta == 0.0:
                continue
            if ref_delta * cand_delta < 0.0:
                count += 1
    return count


def compute_metrics(
    candidate_name: str,
    weights: tuple[float, float, float],
    df: pd.DataFrame,
    subset: str,
) -> MetricRow:
    reference = df["full_reference_aesthetic_score"].to_numpy(float)
    candidate = candidate_scores(df, weights)
    diff = candidate - reference
    nan_mask = np.isnan(candidate) | np.isnan(reference)
    range_errors = int(np.sum((candidate < 0.0) | (candidate > 1.0)))
    valid = ~nan_mask
    ref_valid = reference[valid]
    cand_valid = candidate[valid]

    top1 = bool(np.argmax(ref_valid) == np.argmax(cand_valid)) if len(ref_valid) else False
    top3, top3_frac = top_overlap(ref_valid, cand_valid, 3)
    top5, top5_frac = top_overlap(ref_valid, cand_valid, 5)
    top10, top10_frac = top_overlap(ref_valid, cand_valid, 10)

    return MetricRow(
        candidate_name=candidate_name,
        nima_weight=weights[0],
        rgnet_weight=weights[1],
        alamp_weight=weights[2],
        subset=subset,
        image_count=int(len(df)),
        mae=float(np.mean(np.abs(diff[valid]))) if len(ref_valid) else np.nan,
        maxae=float(np.max(np.abs(diff[valid]))) if len(ref_valid) else np.nan,
        rmse=float(np.sqrt(np.mean(np.square(diff[valid])))) if len(ref_valid) else np.nan,
        spearman=spearman_corr(ref_valid, cand_valid),
        pearson=correlation(ref_valid, cand_valid),
        top1_match=top1,
        top3_overlap=top3,
        top3_overlap_fraction=top3_frac,
        top5_overlap=top5,
        top5_overlap_fraction=top5_frac,
        top10_overlap=top10,
        top10_overlap_fraction=top10_frac,
        ranking_inversions=inversion_count(ref_valid, cand_valid),
        nan_count=int(np.sum(nan_mask)),
        range_error_count=range_errors,
    )


def metric_row_to_dict(row: MetricRow) -> dict[str, object]:
    return row.__dict__.copy()


def grid_weights(step: float) -> list[tuple[float, float, float]]:
    units = int(round(1.0 / step))
    weights = []
    for n_units in range(units + 1):
        for r_units in range(units - n_units + 1):
            a_units = units - n_units - r_units
            weights.append((n_units / units, r_units / units, a_units / units))
    return weights


def refine_weights(centers: Iterable[tuple[float, float, float]], step: float, radius: float) -> list[tuple[float, float, float]]:
    seen: set[tuple[float, float, float]] = set()
    candidates: list[tuple[float, float, float]] = []
    units = int(round(1.0 / step))
    radius_units = int(round(radius / step))
    for center in centers:
        center_units = tuple(int(round(value * units)) for value in center)
        for n_units in range(max(0, center_units[0] - radius_units), min(units, center_units[0] + radius_units) + 1):
            for r_units in range(max(0, center_units[1] - radius_units), min(units, center_units[1] + radius_units) + 1):
                a_units = units - n_units - r_units
                if a_units < 0:
                    continue
                if abs(a_units - center_units[2]) > radius_units:
                    continue
                weights = (n_units / units, r_units / units, a_units / units)
                rounded = tuple(round(value, 4) for value in weights)
                if rounded not in seen:
                    seen.add(rounded)
                    candidates.append(weights)
    return candidates


def candidate_name(weights: tuple[float, float, float], prefix: str = "w") -> str:
    for name, special_weights in SPECIAL_CANDIDATES.items():
        if all(abs(a - b) < 1e-9 for a, b in zip(weights, special_weights)):
            return name
    return f"{prefix}_{weights[0]:.3f}_{weights[1]:.3f}_{weights[2]:.3f}"


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=[
            "top10_overlap_fraction",
            "top5_overlap_fraction",
            "top3_overlap_fraction",
            "top1_match",
            "spearman",
            "mae",
            "maxae",
            "ranking_inversions",
        ],
        ascending=[False, False, False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def run_weight_search(score_df: pd.DataFrame, grid_step: float, refine_step: float, refine_radius: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    initial_weights = grid_weights(grid_step)
    for weights in SPECIAL_CANDIDATES.values():
        if weights not in initial_weights:
            initial_weights.append(weights)

    initial_rows = [metric_row_to_dict(compute_metrics(candidate_name(weights, "grid"), weights, score_df, "full")) for weights in initial_weights]
    initial_results = sort_results(pd.DataFrame(initial_rows))
    centers = list(
        initial_results.head(10)[["nima_weight", "rgnet_weight", "alamp_weight"]]
        .itertuples(index=False, name=None)
    )
    refined = refine_weights(centers, refine_step, refine_radius)
    all_weights = {tuple(round(v, 4) for v in weights): weights for weights in itertools.chain(initial_weights, refined)}
    all_rows = [metric_row_to_dict(compute_metrics(candidate_name(weights, "refined"), weights, score_df, "full")) for weights in all_weights.values()]
    all_results = sort_results(pd.DataFrame(all_rows))
    top = all_results.head(25).copy()
    return all_results, top


def add_holdout_metrics(score_df: pd.DataFrame, top_results: pd.DataFrame) -> pd.DataFrame:
    if len(score_df) < 40:
        return pd.DataFrame()
    search_df = score_df.iloc[::2].reset_index(drop=True)
    holdout_df = score_df.iloc[1::2].reset_index(drop=True)
    rows = []
    for _, row in top_results.head(10).iterrows():
        weights = (float(row["nima_weight"]), float(row["rgnet_weight"]), float(row["alamp_weight"]))
        rows.append(metric_row_to_dict(compute_metrics(str(row["candidate_name"]), weights, search_df, "search_even_index")))
        rows.append(metric_row_to_dict(compute_metrics(str(row["candidate_name"]), weights, holdout_df, "holdout_odd_index")))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:
    display = df.loc[:, columns].head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def format_weights(row: pd.Series) -> str:
    return f"NIMA={row['nima_weight']:.3f}, RGNet={row['rgnet_weight']:.3f}, A-LAMP={row['alamp_weight']:.3f}"


def final_decision(best: pd.Series, proposed: pd.Series, score_df: pd.DataFrame) -> tuple[str, str]:
    if score_df["icaa_score"].isna().any() or score_df[["nima_score", "rgnet_score", "alamp_score"]].isna().any().any():
        return "D", "Model execution or score table construction produced NaNs."
    if int(best["top10_overlap"]) < min(10, len(score_df)) * 0.8 or float(best["spearman"]) < 0.90:
        return "E", "Best available 3-model candidate does not preserve the ICAA-included ranking well enough."
    close_topk = (
        int(proposed["top10_overlap"]) == int(best["top10_overlap"])
        and int(proposed["top5_overlap"]) == int(best["top5_overlap"])
        and int(proposed["top3_overlap"]) == int(best["top3_overlap"])
    )
    close_rank = float(best["spearman"]) - float(proposed["spearman"]) <= 0.001
    close_error = float(proposed["mae"]) - float(best["mae"]) <= 0.002
    if close_topk and close_rank and close_error:
        return "A", "The proposed 0.2/0.3/0.5 ratio is best or close enough under the benchmark criteria."
    return "B", "Another ratio gives a better preservation tradeoff than the proposed ratio."


def write_reports(
    out_dir: Path,
    inventory: pd.DataFrame,
    score_df: pd.DataFrame,
    all_results: pd.DataFrame,
    top_results: pd.DataFrame,
    holdout_results: pd.DataFrame,
    reference_icaa_output: str,
) -> None:
    best = top_results.iloc[0]
    proposed = all_results.loc[all_results["candidate_name"] == "proposed_020_030_050"].iloc[0]
    old = all_results.loc[all_results["candidate_name"] == "old_no_icaa_renormalized"].iloc[0]
    equal = all_results.loc[all_results["candidate_name"] == "equal"].iloc[0]
    decision, decision_reason = final_decision(best, proposed, score_df)

    key_cols = [
        "candidate_name",
        "nima_weight",
        "rgnet_weight",
        "alamp_weight",
        "mae",
        "maxae",
        "rmse",
        "spearman",
        "pearson",
        "top1_match",
        "top3_overlap",
        "top5_overlap",
        "top10_overlap",
        "ranking_inversions",
        "nan_count",
        "range_error_count",
    ]
    compare = pd.DataFrame([best, proposed, old, equal]).drop_duplicates(subset=["candidate_name"])

    report = f"""# A-cut ICAA Skip Weight Search

## 1. Conclusion
- dataset used: `{SOURCE_SCORES_DEFAULT.relative_to(REPO_ROOT)}` plus WSL ICAA SavedModel inference on {len(score_df)} images.
- reference ensemble: NIMA 0.10, RGNet 0.40, A-LAMP 0.30, ICAA 0.20 using ICAA `{reference_icaa_output}` output.
- best candidate: {format_weights(best)}.
- user proposed 0.2/0.3/0.5 result: MAE {proposed['mae']:.6f}, Spearman {proposed['spearman']:.6f}, Top-10 overlap {int(proposed['top10_overlap'])}/10.
- recommendation: Final decision {decision}. {decision_reason}
- whether ready for Flutter implementation: benchmark-backed prompt is generated, but production implementation should still verify the active Flutter branch because the accessible checkout did not contain an ICAA active ensemble.

## 2. Artifact Inventory
Inventory written to `artifact_inventory.csv`. Existing score cache and ICAA SavedModel were available; no cached four-model NIMA/RGNet/A-LAMP/ICAA score table was found, so ICAA was scored in WSL.

## 3. Score Table Construction
The score table reuses cached 50-image NIMA/RGNet/A-LAMP scores from the selected A-cut cache. NIMA was normalized with the same unit-score convention used by the selector: 1..10 scores map to `(score - 1) / 9`, while RGNet and A-LAMP were treated as unit scores. ICAA was computed from the TF-native SavedModel with RGB 224x224 bilinear resize and ImageNet mean/std normalization.

## 4. Reference Definition
Confirmed fact: WSL has an executable ICAA TF-native SavedModel with two outputs, and the local ICAA report names output index 1 as the active color aesthetic output. Assumption for this benchmark: the ICAA-included teacher uses NIMA 0.10, RGNet 0.40, A-LAMP 0.30, ICAA 0.20. The accessible Flutter checkout inspected in read-only mode did not confirm an active ICAA ensemble.

## 5. Weight Search Method
The first pass searched NIMA/RGNet/A-LAMP weights at step 0.05, constrained to non-negative weights summing to 1.0. The search then refined around the top candidates at step 0.01. The proposed 0.20/0.30/0.50, old no-ICAA renormalized 0.125/0.50/0.375, and equal weights were always included.

## 6. Results
{markdown_table(compare, key_cols, max_rows=10)}

Top candidates:

{markdown_table(top_results, key_cols, max_rows=10)}

## 7. Proposed Ratio Analysis
NIMA=0.200, RGNet=0.300, A-LAMP=0.500 produced MAE {proposed['mae']:.6f}, MaxAE {proposed['maxae']:.6f}, RMSE {proposed['rmse']:.6f}, Spearman {proposed['spearman']:.6f}, Pearson {proposed['pearson']:.6f}, Top-1 match {proposed['top1_match']}, Top-3 overlap {int(proposed['top3_overlap'])}/3, Top-5 overlap {int(proposed['top5_overlap'])}/5, and Top-10 overlap {int(proposed['top10_overlap'])}/10.

## 8. Recommended Ratio
Recommended ratio: {format_weights(best)}. It was selected by prioritizing Top-K overlap, then Spearman rank correlation, then MAE/MaxAE, with deterministic tie handling.

## 9. Risks
- ICAA removal changes the aesthetic signal; this benchmark only measures preservation of the selected ICAA-included teacher, not human preference.
- Results reflect the available {len(score_df)}-image WSL test set.
- Android speed gain was supplied from Android logs and was not remeasured here.
- The accessible Flutter checkout did not expose the ICAA skip flag or ICAA active ensemble, so implementation must verify the target branch before changing defaults.

## 10. Next Action
Use `flutter_implementation_prompt.md` for a separate Flutter-stage change only after confirming the target Flutter branch contains the ICAA path and skip flag. Preserve technical scoring and run Flutter validation there.

Final decision: {decision}
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    proposed_report = f"""# Proposed Ratio Analysis

Ratio: NIMA=0.200, RGNet=0.300, A-LAMP=0.500.

{markdown_table(pd.DataFrame([proposed]), key_cols, max_rows=1)}

Decision: {decision}. {decision_reason}
"""
    (out_dir / "proposed_020_030_050_report.md").write_text(proposed_report, encoding="utf-8")

    best_report = f"""# Best Weight Recommendation

Recommended ratio: {format_weights(best)}.

{markdown_table(pd.DataFrame([best]), key_cols, max_rows=1)}

Compared rows:

{markdown_table(compare, key_cols, max_rows=10)}

Final decision: {decision}. {decision_reason}
"""
    (out_dir / "best_weight_recommendation.md").write_text(best_report, encoding="utf-8")

    prompt = f"""# Flutter Implementation Prompt

You are working in the Flutter project. Do not change A-cut technical scoring, KonIQ/FLIVE weights, model assets, or production metadata unrelated to the ICAA skip experiment.

Goal:
- Make ICAA skipped by default for A-cut aesthetic scoring if the target branch has the ICAA path.
- Keep ICAA available behind a debug or override flag if possible.
- Set the non-ICAA aesthetic weights to {format_weights(best)}.
- Preserve all technical scoring behavior.
- Run `flutter analyze`.
- Document the exact files changed and the before/after scoring contract.

Context:
- WSL benchmark output: `{out_dir / 'report.md'}`.
- Benchmark reference assumed NIMA 0.10, RGNet 0.40, A-LAMP 0.30, ICAA 0.20 with ICAA `{reference_icaa_output}` output.
- Accessible Flutter checkout inspected from WSL did not contain the ICAA active ensemble, so first verify the actual target branch implements `POZY_ACUT_SKIP_ICAA_EXPERIMENT` or equivalent before editing.

Strict boundaries:
- Do not modify technical score formulas.
- Do not reintroduce TOPIQ/C6.
- Do not copy or replace model files unless explicitly requested.
- Do not claim production readiness without Flutter-side validation and ranking checks.
"""
    (out_dir / "flutter_implementation_prompt.md").write_text(prompt, encoding="utf-8")

    next_action = f"""# Next Action

Final decision: {decision}. {decision_reason}

1. Review `report.md` and `best_weight_recommendation.md`.
2. Confirm the actual Flutter target branch contains the ICAA ensemble and skip flag.
3. If confirmed, use `flutter_implementation_prompt.md` for the separate Flutter implementation stage.
4. After Flutter changes, run Android parity/ranking checks on the same image set before treating the ratio as production-ready.
"""
    (out_dir / "next_action.md").write_text(next_action, encoding="utf-8")

    if not holdout_results.empty:
        holdout_results.to_csv(out_dir / "holdout_check_metrics.csv", index=False)
        with (out_dir / "report.md").open("a", encoding="utf-8") as fh:
            fh.write("\n## Holdout Check\n")
            fh.write(markdown_table(holdout_results, key_cols + ["subset"], max_rows=20))
            fh.write("\n")

    console = f"""Generated output paths:
- {out_dir / 'report.md'}
- {out_dir / 'artifact_inventory.csv'}
- {out_dir / 'score_table.csv'}
- {out_dir / 'weight_grid_results.csv'}
- {out_dir / 'top_weight_candidates.csv'}
Dataset/image count: {len(score_df)}
Reference weights used: NIMA=0.10, RGNet=0.40, A-LAMP=0.30, ICAA=0.20, ICAA output={reference_icaa_output}
Proposed 0.2/0.3/0.5 metrics: MAE={proposed['mae']:.6f}, MaxAE={proposed['maxae']:.6f}, Spearman={proposed['spearman']:.6f}, Top-10={int(proposed['top10_overlap'])}/10
Best candidate weights and metrics: {format_weights(best)}, MAE={best['mae']:.6f}, MaxAE={best['maxae']:.6f}, Spearman={best['spearman']:.6f}, Top-10={int(best['top10_overlap'])}/10
Final decision {decision}: {decision_reason}
"""
    print(console)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_artifact_inventory(out_dir, args.source_scores_csv.resolve(), args.icaa_saved_model.resolve())
    score_df = load_base_scores(args.source_scores_csv.resolve())
    score_df = filter_readable_images(score_df, out_dir)
    score_df = run_icaa_saved_model(score_df, args.icaa_saved_model.resolve(), args.batch_size)
    score_df = add_reference_and_special_scores(score_df, args.reference_icaa_output)
    score_df.to_csv(out_dir / "score_table.csv", index=False)

    all_results, top_results = run_weight_search(score_df, args.grid_step, args.refine_step, args.refine_radius)
    holdout_results = add_holdout_metrics(score_df, top_results)

    all_results.to_csv(out_dir / "weight_grid_results.csv", index=False)
    top_results.to_csv(out_dir / "top_weight_candidates.csv", index=False)

    write_reports(out_dir, inventory, score_df, all_results, top_results, holdout_results, args.reference_icaa_output)


if __name__ == "__main__":
    main()
