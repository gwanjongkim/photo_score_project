#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = (
    REPO_ROOT / "outputs" / "test_vila_aesthetic_ensemble_experiment_20260423" / "scores_with_ensemble.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "test_vila_aesthetic_only_stage5_current_conservative_20260423"

REQUIRED_COLUMNS = {
    "image_path",
    "stage5_aesthetic_score",
    "aadb_composition",
    "nima_ava_unit",
    "alamp_aadb",
    "rgnet_aadb",
}

CURRENT_WEIGHTS = {
    "rgnet_aadb": 0.30,
    "alamp_aadb": 0.30,
    "aadb_composition": 0.25,
    "nima_ava_unit": 0.15,
}
CONSERVATIVE_WEIGHTS = {
    "rgnet_aadb": 0.35,
    "alamp_aadb": 0.20,
    "aadb_composition": 0.30,
    "nima_ava_unit": 0.15,
}

# Human-auditable visual grouping hints for test_vila only. These are not model labels.
LOW_LIGHT_GROUPS = {
    "1675564423029-24.jpg": "night_city_landmark",
    "1675564423029-9.jpg": "night_city_landmark",
    "IMG_2729.JPG": "night_city_landmark",
    "IMG_2738.JPG": "night_city_landmark",
    "IMG_2739.JPG": "night_city_landmark",
    "IMG_2748.JPG": "night_city_landmark",
    "20250205_000515.jpg": "star_sky_astro_like",
    "1738744324186.jpg": "star_sky_astro_like",
    "1675342165226-3.jpg": "silhouette_dramatic_low_light",
    "1675342165226-5.jpg": "silhouette_dramatic_low_light",
    "1675342165226-13.jpg": "silhouette_dramatic_low_light",
    "20250204_174237(0).jpg": "silhouette_dramatic_low_light",
    "20250205_072644.jpg": "silhouette_dramatic_low_light",
    "20250129_103313.jpg": "silhouette_dramatic_low_light",
    "20230201_181510.jpg": "silhouette_dramatic_low_light",
    "20240515_073118.jpg": "silhouette_dramatic_low_light",
    "IMG_2781.JPG": "silhouette_dramatic_low_light",
    "20250204_170733.jpg": "silhouette_dramatic_low_light",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-production aesthetic-only comparison package for test_vila. "
            "This reuses existing stage5/RGNet/A-Lamp/AADB/NIMA component scores and "
            "does not run model inference or mix technical scores."
        )
    )
    parser.add_argument("--source_csv", default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sheet_limit", type=int, default=20)
    return parser


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_source_frame(source_csv: Path) -> pd.DataFrame:
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")
    frame = pd.read_csv(source_csv)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Source CSV is missing required columns: {', '.join(missing)}")
    return frame


def weighted_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        score = score + weight * pd.to_numeric(frame[column], errors="coerce")
    return score.astype(float)


def rank_desc(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(method="min", ascending=False, na_option="bottom").astype(int)


def score_summary(frame: pd.DataFrame, score_col: str, rank_col: str) -> dict[str, object]:
    values = pd.to_numeric(frame[score_col], errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "max": None, "range": None, "mean": None, "std": None, "top10": []}
    ranked = frame.sort_values([rank_col, "image_file_name"]).head(10)
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "top10": ranked["image_file_name"].tolist(),
    }


def top_set(frame: pd.DataFrame, rank_col: str, top_k: int) -> set[str]:
    return set(frame.loc[frame[rank_col] <= top_k, "image_file_name"])


def overlap_summary(frame: pd.DataFrame, left: str, left_rank: str, right: str, right_rank: str, top_k: int) -> dict[str, object]:
    left_set = top_set(frame, left_rank, top_k)
    right_set = top_set(frame, right_rank, top_k)
    overlap = sorted(left_set & right_set)
    return {
        "top_k": top_k,
        "overlap_count": len(overlap),
        "overlap_images": overlap,
        f"{left}_only": sorted(left_set - right_set),
        f"{right}_only": sorted(right_set - left_set),
    }


def classify_disagreement(row: pd.Series, top_k: int) -> str:
    top_members = []
    if bool(row["in_stage5_top10"]):
        top_members.append("stage5")
    if bool(row["in_current_ensemble_top10"]):
        top_members.append("current_ensemble")
    if bool(row["in_conservative_ensemble_top10"]):
        top_members.append("conservative_ensemble")

    if len(top_members) == 3:
        return "top10_all_three"
    if len(top_members) == 2:
        return "top10_" + "_and_".join(top_members)
    if len(top_members) == 1:
        return "top10_" + top_members[0] + "_only"
    if int(row["max_abs_rank_delta"]) >= max(10, top_k):
        return "large_midpack_rank_disagreement"
    return "minor_or_midpack"


def disagreement_priority(disagreement_type: str) -> int:
    order = {
        "top10_all_three": 10,
        "top10_current_ensemble_and_conservative_ensemble": 20,
        "top10_stage5_and_conservative_ensemble": 25,
        "top10_stage5_and_current_ensemble": 30,
        "top10_conservative_ensemble_only": 35,
        "top10_current_ensemble_only": 40,
        "top10_stage5_only": 45,
        "large_midpack_rank_disagreement": 60,
        "minor_or_midpack": 90,
    }
    return order.get(disagreement_type, 80)


def review_prompt(row: pd.Series) -> str:
    if not bool(row["low_light_focus"]):
        return ""
    group = str(row["low_light_group"])
    if group == "night_city_landmark":
        return "Check whether landmarks/night lights are aesthetically promoted without technical penalties."
    if group == "star_sky_astro_like":
        return "Check whether star/astro mood is valued without rewarding an unclear frame."
    if group == "silhouette_dramatic_low_light":
        return "Check whether silhouette/drama reads as composition rather than only darkness."
    return "Check low-light aesthetic strength separately from technical quality."


def build_comparison_frame(source: pd.DataFrame, top_k: int) -> pd.DataFrame:
    frame = source.copy()
    for column in REQUIRED_COLUMNS - {"image_path"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["image_file_name"] = frame["image_path"].map(lambda value: Path(str(value)).name)
    frame["stage5_student_full_aadb_score"] = frame["stage5_aesthetic_score"].astype(float)
    frame["ensemble_current_score"] = weighted_score(frame, CURRENT_WEIGHTS)
    frame["ensemble_conservative_score"] = weighted_score(frame, CONSERVATIVE_WEIGHTS)

    frame["stage5_rank"] = rank_desc(frame["stage5_student_full_aadb_score"])
    frame["ensemble_current_rank"] = rank_desc(frame["ensemble_current_score"])
    frame["ensemble_conservative_rank"] = rank_desc(frame["ensemble_conservative_score"])

    frame["current_minus_stage5_rank_delta"] = frame["ensemble_current_rank"] - frame["stage5_rank"]
    frame["conservative_minus_stage5_rank_delta"] = frame["ensemble_conservative_rank"] - frame["stage5_rank"]
    frame["conservative_minus_current_rank_delta"] = frame["ensemble_conservative_rank"] - frame["ensemble_current_rank"]
    frame["current_minus_stage5_score_delta"] = frame["ensemble_current_score"] - frame["stage5_student_full_aadb_score"]
    frame["conservative_minus_stage5_score_delta"] = frame["ensemble_conservative_score"] - frame["stage5_student_full_aadb_score"]
    frame["conservative_minus_current_score_delta"] = frame["ensemble_conservative_score"] - frame["ensemble_current_score"]

    frame["max_abs_rank_delta"] = frame[
        [
            "current_minus_stage5_rank_delta",
            "conservative_minus_stage5_rank_delta",
            "conservative_minus_current_rank_delta",
        ]
    ].abs().max(axis=1)
    frame["max_abs_score_delta"] = frame[
        [
            "current_minus_stage5_score_delta",
            "conservative_minus_stage5_score_delta",
            "conservative_minus_current_score_delta",
        ]
    ].abs().max(axis=1)

    frame["in_stage5_top10"] = frame["stage5_rank"] <= top_k
    frame["in_current_ensemble_top10"] = frame["ensemble_current_rank"] <= top_k
    frame["in_conservative_ensemble_top10"] = frame["ensemble_conservative_rank"] <= top_k
    frame["low_light_group"] = frame["image_file_name"].map(LOW_LIGHT_GROUPS).fillna("")
    frame["low_light_focus"] = frame["low_light_group"] != ""
    frame["disagreement_type"] = frame.apply(classify_disagreement, axis=1, top_k=top_k)
    frame["review_priority"] = frame["disagreement_type"].map(disagreement_priority).astype(int)
    frame["dark_low_light_review_prompt"] = frame.apply(review_prompt, axis=1)
    frame["human_pick"] = ""
    frame["human_note"] = ""

    ordered_columns = [
        "image_file_name",
        "image_path",
        "low_light_focus",
        "low_light_group",
        "stage5_student_full_aadb_score",
        "stage5_rank",
        "ensemble_current_score",
        "ensemble_current_rank",
        "ensemble_conservative_score",
        "ensemble_conservative_rank",
        "current_minus_stage5_rank_delta",
        "conservative_minus_stage5_rank_delta",
        "conservative_minus_current_rank_delta",
        "current_minus_stage5_score_delta",
        "conservative_minus_stage5_score_delta",
        "conservative_minus_current_score_delta",
        "max_abs_rank_delta",
        "max_abs_score_delta",
        "in_stage5_top10",
        "in_current_ensemble_top10",
        "in_conservative_ensemble_top10",
        "disagreement_type",
        "review_priority",
        "aadb_composition",
        "nima_ava_unit",
        "alamp_aadb",
        "rgnet_aadb",
        "dark_low_light_review_prompt",
        "human_pick",
        "human_note",
    ]
    return frame[ordered_columns].sort_values(
        ["review_priority", "max_abs_rank_delta", "stage5_rank", "image_file_name"],
        ascending=[True, False, True, True],
    )


def write_csv_and_json(frame: pd.DataFrame, csv_path: Path, json_path: Path | None = None) -> None:
    frame.to_csv(csv_path, index=False)
    if json_path is not None:
        json_path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")


def sorted_top(frame: pd.DataFrame, rank_col: str, top_k: int) -> pd.DataFrame:
    return frame.sort_values([rank_col, "image_file_name"]).head(top_k)


def sheet_label(row: pd.Series) -> str:
    group = str(row["low_light_group"]) if bool(row["low_light_focus"]) else "general"
    return (
        f"{row['image_file_name']}\n"
        f"s5 #{int(row['stage5_rank'])} {float(row['stage5_student_full_aadb_score']):.3f} | "
        f"cur #{int(row['ensemble_current_rank'])} {float(row['ensemble_current_score']):.3f}\n"
        f"con #{int(row['ensemble_conservative_rank'])} {float(row['ensemble_conservative_score']):.3f} | {group}"
    )


def load_thumbnail(path_value: str, size: tuple[int, int]) -> Image.Image:
    image_path = resolve_repo_path(path_value)
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def collect_render_warnings(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        image_path_value = str(row["image_path"])
        if image_path_value in seen:
            continue
        seen.add(image_path_value)
        image_path = resolve_repo_path(image_path_value)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            rows.append(
                {
                    "image_file_name": row["image_file_name"],
                    "image_path": image_path_value,
                    "render_warning": str(exc),
                }
            )
    return pd.DataFrame(rows, columns=["image_file_name", "image_path", "render_warning"])


def make_contact_sheet(
    rows: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    limit: int,
    columns: int = 5,
) -> None:
    rows = rows.head(limit).reset_index(drop=True)
    thumb_w, thumb_h = 240, 180
    label_h, title_h, pad = 74, 36, 8
    grid_rows = max(1, math.ceil(max(1, len(rows)) / columns))
    canvas = Image.new(
        "RGB",
        (columns * thumb_w + (columns + 1) * pad, title_h + grid_rows * (thumb_h + label_h + pad) + pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 8), title, fill="black", font=font)

    if rows.empty:
        draw.text((pad, title_h + pad), "No matching images.", fill="black", font=font)
        canvas.save(output_path, quality=92)
        return

    for index, row in rows.iterrows():
        col = index % columns
        grid_row = index // columns
        x = pad + col * (thumb_w + pad)
        y = title_h + pad + grid_row * (thumb_h + label_h + pad)
        try:
            thumb = load_thumbnail(str(row["image_path"]), (thumb_w, thumb_h))
        except Exception as exc:  # pragma: no cover - visual report should survive a missing image
            thumb = Image.new("RGB", (thumb_w, thumb_h), "#eeeeee")
            ImageDraw.Draw(thumb).text((8, 8), f"Image error:\n{exc}", fill="black", font=font)
        canvas.paste(thumb, (x, y))
        draw.multiline_text((x, y + thumb_h + 4), sheet_label(row), fill="black", font=font, spacing=2)

    canvas.save(output_path, quality=92)


def write_review_index(output_dir: Path, summary: dict[str, object]) -> None:
    contacts = [
        ("Stage5 aesthetic-only top 10", "contact_01_stage5_aesthetic_only_top10.jpg"),
        ("Current ensemble aesthetic-only top 10", "contact_02_current_ensemble_aesthetic_only_top10.jpg"),
        ("Conservative ensemble aesthetic-only top 10", "contact_03_conservative_ensemble_aesthetic_only_top10.jpg"),
        ("Top-10 and large disagreement cases", "contact_04_major_disagreement_cases.jpg"),
        ("Dark / night / silhouette focus cases", "contact_05_dark_low_light_focus_cases.jpg"),
        ("Current vs conservative disagreement cases", "contact_06_current_vs_conservative_disagreements.jpg"),
        ("Stage5 under-ranks low-light candidates", "contact_07_stage5_under_ranks_low_light.jpg"),
    ]
    stats_rows = []
    for label, stats in summary["score_summary"].items():
        stats_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{stats['min']:.4f}</td>"
            f"<td>{stats['max']:.4f}</td>"
            f"<td>{stats['range']:.4f}</td>"
            f"<td>{stats['mean']:.4f}</td>"
            f"<td>{stats['std']:.4f}</td>"
            "</tr>"
        )
    contact_html = "\n".join(
        f"<section><h2>{html.escape(title)}</h2><img src=\"{html.escape(path)}\" alt=\"{html.escape(title)}\"></section>"
        for title, path in contacts
    )
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Aesthetic-only comparison: stage5 vs current vs conservative</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111; }}
    table {{ border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    code {{ background: #f3f3f3; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>Aesthetic-only comparison</h1>
  <p>This package uses only stage5, RGNet, A-Lamp, AADB, and NIMA aesthetic scores. It does not use technical score, final production mixing, gates, or penalties.</p>
  <p>Open <code>aesthetic_only_comparison.csv</code> for manual notes, then inspect the contact sheets below.</p>
  <table>
    <thead><tr><th>Candidate</th><th>Min</th><th>Max</th><th>Range</th><th>Mean</th><th>Std</th></tr></thead>
    <tbody>{''.join(stats_rows)}</tbody>
  </table>
  {contact_html}
</body>
</html>
"""
    (output_dir / "review_index.html").write_text(index, encoding="utf-8")


def write_readme(output_dir: Path, summary: dict[str, object]) -> None:
    render_warning_note = ""
    if summary.get("contact_sheet_render_warnings"):
        render_warning_note = (
            "\n## Contact sheet render warnings\n\n"
            "Some scored files could not be thumbnailed by Pillow for contact sheets. "
            "See `contact_sheet_render_warnings.csv`; scores and CSV rows are still present.\n"
        )
    readme = f"""# Aesthetic-only comparison: stage5 vs current vs conservative

Dataset: `{summary["input_dir"]}`

Source score CSV: `{summary["source_csv"]}`

This is a non-production review package. It reuses existing scores and ranks images using aesthetic scores only.

No technical quality fields are used for ranking here. The prior source CSV contains technical/final columns, but this package ignores them.

## Formulas

- `stage5_student_full_aadb`: `stage5_aesthetic_score`
- `ensemble_current`: `0.30 * rgnet_aadb + 0.30 * alamp_aadb + 0.25 * aadb_composition + 0.15 * nima_ava_unit`
- `ensemble_conservative`: `0.35 * rgnet_aadb + 0.20 * alamp_aadb + 0.30 * aadb_composition + 0.15 * nima_ava_unit`

Component normalization is inherited from the prior experiment:

- AADB, A-Lamp, and RGNet are clipped unit scores.
- NIMA unit score is `(nima_mean_score - 1.0) / 9.0`, clipped to `[0, 1]`.

## Suggested review order

1. Open `review_index.html`.
2. Compare `contact_01_stage5_aesthetic_only_top10.jpg`, `contact_02_current_ensemble_aesthetic_only_top10.jpg`, and `contact_03_conservative_ensemble_aesthetic_only_top10.jpg`.
3. Review `contact_04_major_disagreement_cases.jpg`.
4. Review `contact_05_dark_low_light_focus_cases.jpg`.
5. Use `aesthetic_only_comparison.csv` to mark `human_pick` and `human_note`.

## Dark / low-light focus

The `low_light_group` column is a human-auditable grouping hint for this `test_vila` set only. It is meant to help review night, star-sky, silhouette, and mood-heavy frames without mixing in technical quality.

Key question: does the conservative ensemble keep visually strong dark/silhouette images high while reducing any A-Lamp-driven over-reward of mood-heavy darkness?
{render_warning_note}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def make_outputs(
    frame: pd.DataFrame,
    *,
    source_csv: Path,
    output_dir: Path,
    top_k: int,
    sheet_limit: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv_and_json(frame, output_dir / "aesthetic_only_comparison.csv", output_dir / "aesthetic_only_comparison.json")
    render_warnings = collect_render_warnings(frame)
    write_csv_and_json(render_warnings, output_dir / "contact_sheet_render_warnings.csv")

    stage5_top = sorted_top(frame, "stage5_rank", top_k)
    current_top = sorted_top(frame, "ensemble_current_rank", top_k)
    conservative_top = sorted_top(frame, "ensemble_conservative_rank", top_k)
    write_csv_and_json(stage5_top, output_dir / "stage5_aesthetic_only_top10.csv")
    write_csv_and_json(current_top, output_dir / "ensemble_current_aesthetic_only_top10.csv")
    write_csv_and_json(conservative_top, output_dir / "ensemble_conservative_aesthetic_only_top10.csv")

    overlap_specs = [
        (
            "stage5_vs_current",
            "stage5_rank",
            "ensemble_current_rank",
            "top10_overlap_stage5_vs_current.csv",
        ),
        (
            "stage5_vs_conservative",
            "stage5_rank",
            "ensemble_conservative_rank",
            "top10_overlap_stage5_vs_conservative.csv",
        ),
        (
            "current_vs_conservative",
            "ensemble_current_rank",
            "ensemble_conservative_rank",
            "top10_overlap_current_vs_conservative.csv",
        ),
    ]
    for _, left_rank, right_rank, file_name in overlap_specs:
        left_set = top_set(frame, left_rank, top_k)
        right_set = top_set(frame, right_rank, top_k)
        overlap = frame[frame["image_file_name"].isin(left_set & right_set)].sort_values(
            [left_rank, right_rank, "image_file_name"]
        )
        write_csv_and_json(overlap, output_dir / file_name)

    all_three = (
        frame[frame["in_stage5_top10"] & frame["in_current_ensemble_top10"] & frame["in_conservative_ensemble_top10"]]
        .sort_values(["stage5_rank", "ensemble_current_rank", "ensemble_conservative_rank", "image_file_name"])
    )
    write_csv_and_json(all_three, output_dir / "top10_overlap_all_three.csv")

    disagreement_cases = frame[
        (frame["disagreement_type"] != "top10_all_three")
        & (
            frame["in_stage5_top10"]
            | frame["in_current_ensemble_top10"]
            | frame["in_conservative_ensemble_top10"]
            | (frame["max_abs_rank_delta"] >= max(10, top_k))
        )
    ].sort_values(["max_abs_rank_delta", "max_abs_score_delta", "stage5_rank"], ascending=[False, False, True])
    write_csv_and_json(disagreement_cases, output_dir / "disagreement_cases.csv")

    dark_cases = frame[frame["low_light_focus"]].sort_values(
        ["max_abs_rank_delta", "ensemble_conservative_rank", "ensemble_current_rank"],
        ascending=[False, True, True],
    )
    write_csv_and_json(dark_cases, output_dir / "dark_low_light_focus_cases.csv")

    dark_group_summary = (
        dark_cases.groupby("low_light_group", dropna=False)
        .agg(
            image_count=("image_file_name", "count"),
            stage5_mean=("stage5_student_full_aadb_score", "mean"),
            current_mean=("ensemble_current_score", "mean"),
            conservative_mean=("ensemble_conservative_score", "mean"),
            stage5_best_rank=("stage5_rank", "min"),
            current_best_rank=("ensemble_current_rank", "min"),
            conservative_best_rank=("ensemble_conservative_rank", "min"),
            stage5_top10_count=("in_stage5_top10", "sum"),
            current_top10_count=("in_current_ensemble_top10", "sum"),
            conservative_top10_count=("in_conservative_ensemble_top10", "sum"),
        )
        .reset_index()
        .sort_values("low_light_group")
    )
    write_csv_and_json(dark_group_summary, output_dir / "dark_low_light_group_summary.csv")

    current_vs_conservative = frame.sort_values(
        ["conservative_minus_current_rank_delta", "max_abs_score_delta"],
        key=lambda column: column.abs() if column.name == "conservative_minus_current_rank_delta" else column,
        ascending=[False, False],
    )
    current_vs_conservative = current_vs_conservative.head(sheet_limit)
    write_csv_and_json(current_vs_conservative, output_dir / "current_vs_conservative_disagreements.csv")

    stage5_under_low_light = dark_cases[
        (dark_cases["current_minus_stage5_rank_delta"] <= -5)
        | (dark_cases["conservative_minus_stage5_rank_delta"] <= -5)
    ].sort_values(["stage5_rank", "max_abs_rank_delta"], ascending=[False, False])
    write_csv_and_json(stage5_under_low_light, output_dir / "stage5_under_ranks_low_light_candidates.csv")

    make_contact_sheet(
        stage5_top,
        title="Stage5 aesthetic-only top 10",
        output_path=output_dir / "contact_01_stage5_aesthetic_only_top10.jpg",
        limit=top_k,
    )
    make_contact_sheet(
        current_top,
        title="Current weighted ensemble aesthetic-only top 10",
        output_path=output_dir / "contact_02_current_ensemble_aesthetic_only_top10.jpg",
        limit=top_k,
    )
    make_contact_sheet(
        conservative_top,
        title="Conservative ensemble aesthetic-only top 10",
        output_path=output_dir / "contact_03_conservative_ensemble_aesthetic_only_top10.jpg",
        limit=top_k,
    )
    make_contact_sheet(
        disagreement_cases,
        title="Major aesthetic-only disagreement cases",
        output_path=output_dir / "contact_04_major_disagreement_cases.jpg",
        limit=sheet_limit,
    )
    make_contact_sheet(
        dark_cases,
        title="Dark / night / silhouette focus cases",
        output_path=output_dir / "contact_05_dark_low_light_focus_cases.jpg",
        limit=sheet_limit,
    )
    make_contact_sheet(
        current_vs_conservative,
        title="Current vs conservative ensemble disagreements",
        output_path=output_dir / "contact_06_current_vs_conservative_disagreements.jpg",
        limit=sheet_limit,
    )
    make_contact_sheet(
        stage5_under_low_light,
        title="Stage5 under-ranks low-light candidates",
        output_path=output_dir / "contact_07_stage5_under_ranks_low_light.jpg",
        limit=sheet_limit,
    )

    summary = {
        "experiment_name": "aesthetic_only_stage5_current_conservative",
        "non_production": True,
        "heavy_rescoring": False,
        "input_dir": str(REPO_ROOT / "test_vila"),
        "source_csv": str(source_csv),
        "output_dir": str(output_dir),
        "num_images": int(len(frame)),
        "top_k": top_k,
        "technical_score_used": False,
        "final_production_mix_used": False,
        "technical_gate_or_penalty_used": False,
        "formulas": {
            "stage5_student_full_aadb": "stage5_aesthetic_score",
            "ensemble_current": "0.30 * rgnet_aadb + 0.30 * alamp_aadb + 0.25 * aadb_composition + 0.15 * nima_ava_unit",
            "ensemble_conservative": "0.35 * rgnet_aadb + 0.20 * alamp_aadb + 0.30 * aadb_composition + 0.15 * nima_ava_unit",
        },
        "normalization_notes": {
            "aadb_composition": "clipped unit score from prior experiment",
            "alamp_aadb": "clipped unit score from prior experiment",
            "rgnet_aadb": "clipped unit score from prior experiment",
            "nima_ava_unit": "clipped (nima_mean_score - 1.0) / 9.0 from prior experiment",
        },
        "score_summary": {
            "stage5_student_full_aadb": score_summary(frame, "stage5_student_full_aadb_score", "stage5_rank"),
            "ensemble_current": score_summary(frame, "ensemble_current_score", "ensemble_current_rank"),
            "ensemble_conservative": score_summary(frame, "ensemble_conservative_score", "ensemble_conservative_rank"),
        },
        "top10_overlap": {
            "stage5_vs_current": overlap_summary(
                frame, "stage5", "stage5_rank", "current", "ensemble_current_rank", top_k
            ),
            "stage5_vs_conservative": overlap_summary(
                frame, "stage5", "stage5_rank", "conservative", "ensemble_conservative_rank", top_k
            ),
            "current_vs_conservative": overlap_summary(
                frame, "current", "ensemble_current_rank", "conservative", "ensemble_conservative_rank", top_k
            ),
            "all_three_count": int(len(all_three)),
            "all_three_images": all_three["image_file_name"].tolist(),
        },
        "dark_low_light": {
            "grouping_note": "Human-auditable test_vila visual grouping hints, not model labels.",
            "num_focus_images": int(len(dark_cases)),
            "group_summary": json.loads(dark_group_summary.to_json(orient="records")),
            "stage5_under_ranked_candidates": stage5_under_low_light["image_file_name"].tolist(),
        },
        "contact_sheet_render_warnings": json.loads(render_warnings.to_json(orient="records")),
        "artifacts": [],
    }
    write_readme(output_dir, summary)
    write_review_index(output_dir, summary)
    summary["artifacts"] = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = build_parser().parse_args()
    source_csv = resolve_repo_path(args.source_csv)
    output_dir = resolve_repo_path(args.output_dir)

    source = load_source_frame(source_csv)
    frame = build_comparison_frame(source, top_k=args.top_k)
    summary = make_outputs(
        frame,
        source_csv=source_csv,
        output_dir=output_dir,
        top_k=args.top_k,
        sheet_limit=args.sheet_limit,
    )

    print(json.dumps({"output_dir": str(output_dir), "num_images": summary["num_images"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
