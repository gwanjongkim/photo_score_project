#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = REPO_ROOT / "outputs" / "test_vila_aesthetic_ensemble_experiment_20260423" / "scores_with_ensemble.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "test_vila_human_acut_review_stage5_vs_ensemble_20260423"

REQUIRED_COLUMNS = {
    "image_path",
    "stage5_aesthetic_score",
    "weighted_aesthetic_ensemble",
    "stage5_current_final_score",
    "ensemble_experiment_final_score",
    "rank_stage5_current_final",
    "rank_ensemble_experiment_final",
    "rank_stage5_aesthetic",
    "rank_ensemble_aesthetic",
    "final_rank_delta_ensemble_minus_stage5",
    "aesthetic_rank_delta_ensemble_minus_stage5",
    "aadb_composition",
    "nima_ava_unit",
    "alamp_aadb",
    "rgnet_aadb",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a human A-cut review package comparing current stage5 picks against "
            "the non-production weighted aesthetic ensemble. This reads existing experiment "
            "outputs and does not run model inference."
        )
    )
    parser.add_argument("--source_csv", default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sheet_limit", type=int, default=10)
    return parser


def resolve_path(path_value: str | Path) -> Path:
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


def classify_disagreement(row: pd.Series, top_k: int) -> str:
    in_stage5_top = bool(row["in_stage5_top10"])
    in_ensemble_top = bool(row["in_ensemble_top10"])
    rank_delta = int(row["rank_difference"])
    aesthetic_delta = int(row["aesthetic_rank_difference"])
    if in_stage5_top and in_ensemble_top:
        return "top10_overlap"
    if in_stage5_top and not in_ensemble_top:
        return "stage5_only_top10"
    if in_ensemble_top and not in_stage5_top:
        return "ensemble_only_top10"
    threshold = max(5, top_k)
    if rank_delta <= -threshold or aesthetic_delta <= -threshold:
        return "ensemble_much_higher"
    if rank_delta >= threshold or aesthetic_delta >= threshold:
        return "stage5_much_higher"
    return "minor_or_midpack"


def classify_bottom_disagreement(row: pd.Series, top_k: int) -> str:
    in_stage5_bottom = bool(row["in_stage5_bottom10"])
    in_ensemble_bottom = bool(row["in_ensemble_bottom10"])
    rank_delta = int(row["rank_difference"])
    aesthetic_delta = int(row["aesthetic_rank_difference"])
    if in_stage5_bottom and in_ensemble_bottom:
        return "bottom10_overlap"
    if in_stage5_bottom and not in_ensemble_bottom:
        return "stage5_only_bottom10"
    if in_ensemble_bottom and not in_stage5_bottom:
        return "ensemble_only_bottom10"
    threshold = max(5, top_k)
    if rank_delta >= threshold or aesthetic_delta >= threshold:
        return "ensemble_much_lower_than_stage5"
    if rank_delta <= -threshold or aesthetic_delta <= -threshold:
        return "stage5_much_lower_than_ensemble"
    return "not_bottom10"


def review_priority(disagreement_type: str) -> int:
    order = {
        "top10_overlap": 10,
        "ensemble_only_top10": 20,
        "stage5_only_top10": 30,
        "ensemble_much_higher": 40,
        "stage5_much_higher": 50,
        "minor_or_midpack": 90,
    }
    return order.get(disagreement_type, 99)


def bottom_review_priority(disagreement_type: str) -> int:
    order = {
        "bottom10_overlap": 10,
        "ensemble_only_bottom10": 20,
        "stage5_only_bottom10": 30,
        "ensemble_much_lower_than_stage5": 40,
        "stage5_much_lower_than_ensemble": 50,
        "not_bottom10": 90,
    }
    return order.get(disagreement_type, 99)


def build_review_frame(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    bottom_rank_start = max(1, len(frame) - top_k + 1)
    review = pd.DataFrame(
        {
            "image_file_name": frame["image_path"].map(lambda value: Path(str(value)).name),
            "image_path": frame["image_path"],
            "stage5_aesthetic_score": frame["stage5_aesthetic_score"],
            "stage5_final_score": frame["stage5_current_final_score"],
            "stage5_final_rank": frame["rank_stage5_current_final"].astype(int),
            "stage5_aesthetic_rank": frame["rank_stage5_aesthetic"].astype(int),
            "ensemble_aesthetic_score": frame["weighted_aesthetic_ensemble"],
            "ensemble_final_score": frame["ensemble_experiment_final_score"],
            "ensemble_final_rank": frame["rank_ensemble_experiment_final"].astype(int),
            "ensemble_aesthetic_rank": frame["rank_ensemble_aesthetic"].astype(int),
            "rank_difference": frame["final_rank_delta_ensemble_minus_stage5"].astype(int),
            "aesthetic_rank_difference": frame["aesthetic_rank_delta_ensemble_minus_stage5"].astype(int),
            "stage5_minus_ensemble_aesthetic_score": frame["stage5_aesthetic_score"] - frame["weighted_aesthetic_ensemble"],
            "in_stage5_top10": frame["rank_stage5_current_final"].astype(int) <= top_k,
            "in_ensemble_top10": frame["rank_ensemble_experiment_final"].astype(int) <= top_k,
            "in_stage5_bottom10": frame["rank_stage5_current_final"].astype(int) >= bottom_rank_start,
            "in_ensemble_bottom10": frame["rank_ensemble_experiment_final"].astype(int) >= bottom_rank_start,
            "aadb_composition": frame["aadb_composition"],
            "nima_ava_unit": frame["nima_ava_unit"],
            "alamp_aadb": frame["alamp_aadb"],
            "rgnet_aadb": frame["rgnet_aadb"],
        }
    )
    review["disagreement_type"] = review.apply(classify_disagreement, axis=1, top_k=top_k)
    review["bottom_disagreement_type"] = review.apply(classify_bottom_disagreement, axis=1, top_k=top_k)
    review["review_priority"] = review["disagreement_type"].map(review_priority).astype(int)
    review["bottom_review_priority"] = review["bottom_disagreement_type"].map(bottom_review_priority).astype(int)
    review["human_pick"] = ""
    review["human_note"] = ""
    review["human_reject_pick"] = ""
    review["human_reject_note"] = ""
    return review.sort_values(
        [
            "review_priority",
            "bottom_review_priority",
            "in_ensemble_top10",
            "in_stage5_top10",
            "rank_difference",
            "stage5_final_rank",
        ],
        ascending=[True, True, False, False, True, True],
    ).reset_index(drop=True)


def sheet_label(row: pd.Series) -> str:
    return (
        f"{Path(str(row['image_path'])).name}\n"
        f"s5 #{int(row['stage5_final_rank'])} {float(row['stage5_final_score']):.3f} | "
        f"ens #{int(row['ensemble_final_rank'])} {float(row['ensemble_final_score']):.3f}\n"
        f"s5_a {float(row['stage5_aesthetic_score']):.3f} | ens_a {float(row['ensemble_aesthetic_score']):.3f}"
    )


def load_thumbnail(path_value: str, size: tuple[int, int]) -> Image.Image:
    path = resolve_path(path_value)
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


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
    label_h, title_h, pad = 62, 36, 8
    grid_rows = max(1, math.ceil(len(rows) / columns))
    canvas = Image.new(
        "RGB",
        (columns * thumb_w + (columns + 1) * pad, title_h + grid_rows * (thumb_h + label_h + pad) + pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 8), title, fill="black", font=font)

    for index, row in rows.iterrows():
        col = index % columns
        grid_row = index // columns
        x = pad + col * (thumb_w + pad)
        y = title_h + grid_row * (thumb_h + label_h + pad)
        try:
            tile = load_thumbnail(str(row["image_path"]), (thumb_w, thumb_h))
        except Exception as exc:
            tile = Image.new("RGB", (thumb_w, thumb_h), "#eeeeee")
            ImageDraw.Draw(tile).text((8, 8), f"Could not load image\n{exc}", fill="black", font=font)
        canvas.paste(tile, (x, y))
        draw.text((x, y + thumb_h + 3), sheet_label(row), fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def write_contact_sheets(review: pd.DataFrame, output_dir: Path, top_k: int, sheet_limit: int) -> dict[str, str]:
    stage5_top = review.sort_values("stage5_final_rank").head(top_k)
    ensemble_top = review.sort_values("ensemble_final_rank").head(top_k)
    stage5_bottom = review.sort_values("stage5_final_rank", ascending=False).head(top_k)
    ensemble_bottom = review.sort_values("ensemble_final_rank", ascending=False).head(top_k)
    overlap = review[review["in_stage5_top10"] & review["in_ensemble_top10"]].sort_values("stage5_final_rank")
    bottom_overlap = review[review["in_stage5_bottom10"] & review["in_ensemble_bottom10"]].sort_values(
        "stage5_final_rank",
        ascending=False,
    )
    top10_disagreements = review[
        review["disagreement_type"].isin(["stage5_only_top10", "ensemble_only_top10"])
    ].sort_values(["disagreement_type", "rank_difference"])
    bottom10_disagreements = review[
        review["bottom_disagreement_type"].isin(["stage5_only_bottom10", "ensemble_only_bottom10"])
    ].sort_values(["bottom_disagreement_type", "rank_difference"], ascending=[True, False])
    ensemble_higher = review.sort_values("rank_difference", ascending=True)
    stage5_higher = review.sort_values("rank_difference", ascending=False)
    ensemble_lower = review.sort_values("rank_difference", ascending=False)
    stage5_lower = review.sort_values("rank_difference", ascending=True)

    sheets = {
        "stage5_top10": "contact_01_stage5_top10.jpg",
        "ensemble_top10": "contact_02_ensemble_top10.jpg",
        "top10_overlap": "contact_03_top10_overlap.jpg",
        "top10_disagreements": "contact_04_top10_disagreements.jpg",
        "ensemble_much_higher": "contact_05_ensemble_much_higher_than_stage5.jpg",
        "stage5_much_higher": "contact_06_stage5_much_higher_than_ensemble.jpg",
        "stage5_bottom10": "contact_07_stage5_bottom10.jpg",
        "ensemble_bottom10": "contact_08_ensemble_bottom10.jpg",
        "bottom10_overlap": "contact_09_bottom10_overlap.jpg",
        "bottom10_disagreements": "contact_10_bottom10_disagreements.jpg",
        "ensemble_much_lower": "contact_11_ensemble_much_lower_than_stage5.jpg",
        "stage5_much_lower": "contact_12_stage5_much_lower_than_ensemble.jpg",
    }
    make_contact_sheet(stage5_top, title="Stage5 current top 10", output_path=output_dir / sheets["stage5_top10"], limit=top_k)
    make_contact_sheet(ensemble_top, title="Weighted ensemble top 10", output_path=output_dir / sheets["ensemble_top10"], limit=top_k)
    make_contact_sheet(overlap, title="Top-10 overlap", output_path=output_dir / sheets["top10_overlap"], limit=max(1, len(overlap)))
    make_contact_sheet(
        top10_disagreements,
        title="Top-10 disagreement cases",
        output_path=output_dir / sheets["top10_disagreements"],
        limit=max(1, min(sheet_limit, len(top10_disagreements))),
    )
    make_contact_sheet(
        ensemble_higher,
        title="Ensemble ranks much higher",
        output_path=output_dir / sheets["ensemble_much_higher"],
        limit=sheet_limit,
    )
    make_contact_sheet(
        stage5_higher,
        title="Stage5 ranks much higher",
        output_path=output_dir / sheets["stage5_much_higher"],
        limit=sheet_limit,
    )
    make_contact_sheet(
        stage5_bottom,
        title="Stage5 bottom 10",
        output_path=output_dir / sheets["stage5_bottom10"],
        limit=top_k,
    )
    make_contact_sheet(
        ensemble_bottom,
        title="Weighted ensemble bottom 10",
        output_path=output_dir / sheets["ensemble_bottom10"],
        limit=top_k,
    )
    make_contact_sheet(
        bottom_overlap,
        title="Bottom-10 overlap",
        output_path=output_dir / sheets["bottom10_overlap"],
        limit=max(1, len(bottom_overlap)),
    )
    make_contact_sheet(
        bottom10_disagreements,
        title="Bottom-10 disagreement cases",
        output_path=output_dir / sheets["bottom10_disagreements"],
        limit=max(1, min(sheet_limit, len(bottom10_disagreements))),
    )
    make_contact_sheet(
        ensemble_lower,
        title="Ensemble ranks much lower",
        output_path=output_dir / sheets["ensemble_much_lower"],
        limit=sheet_limit,
    )
    make_contact_sheet(
        stage5_lower,
        title="Stage5 ranks much lower",
        output_path=output_dir / sheets["stage5_much_lower"],
        limit=sheet_limit,
    )
    return sheets


def safe_json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


def write_text_outputs(review: pd.DataFrame, output_dir: Path, source_csv: Path, top_k: int, sheets: dict[str, str]) -> None:
    stage5_top = review[review["in_stage5_top10"]]
    ensemble_top = review[review["in_ensemble_top10"]]
    overlap = review[review["in_stage5_top10"] & review["in_ensemble_top10"]]
    stage5_only = review[review["disagreement_type"] == "stage5_only_top10"]
    ensemble_only = review[review["disagreement_type"] == "ensemble_only_top10"]
    stage5_bottom = review[review["in_stage5_bottom10"]]
    ensemble_bottom = review[review["in_ensemble_bottom10"]]
    bottom_overlap = review[review["in_stage5_bottom10"] & review["in_ensemble_bottom10"]]
    stage5_only_bottom = review[review["bottom_disagreement_type"] == "stage5_only_bottom10"]
    ensemble_only_bottom = review[review["bottom_disagreement_type"] == "ensemble_only_bottom10"]

    summary = {
        "package": "stage5_vs_weighted_ensemble_human_acut_review",
        "source_csv": str(source_csv),
        "top_k": int(top_k),
        "num_images": int(len(review)),
        "stage5_top10_count": int(len(stage5_top)),
        "ensemble_top10_count": int(len(ensemble_top)),
        "top10_overlap_count": int(len(overlap)),
        "stage5_only_top10": stage5_only["image_file_name"].tolist(),
        "ensemble_only_top10": ensemble_only["image_file_name"].tolist(),
        "stage5_bottom10_count": int(len(stage5_bottom)),
        "ensemble_bottom10_count": int(len(ensemble_bottom)),
        "bottom10_overlap_count": int(len(bottom_overlap)),
        "stage5_only_bottom10": stage5_only_bottom["image_file_name"].tolist(),
        "ensemble_only_bottom10": ensemble_only_bottom["image_file_name"].tolist(),
        "rank_difference_convention": "rank_difference = ensemble_final_rank - stage5_final_rank; negative means ensemble ranks the image higher.",
        "bottom_rank_convention": "Bottom-10 uses largest final-rank numbers. Positive rank_difference means the ensemble ranks the image lower; negative means stage5 ranks it lower.",
        "contact_sheets": sheets,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = f"""# Human A-cut Review: Stage5 vs Weighted Ensemble

This package reuses existing scores from:

`{source_csv}`

No model scoring or retraining was run to create this package.

## What To Open First

1. `review_index.html` for a visual overview.
2. `human_acut_comparison_sheet.csv` when you want to record manual choices.

## How To Judge

- Compare `contact_01_stage5_top10.jpg` and `contact_02_ensemble_top10.jpg`.
- Use `contact_03_top10_overlap.jpg` to see what both paths agree on.
- Use `contact_04_top10_disagreements.jpg` to inspect the images that only one top-10 list selected.
- Use `contact_05_ensemble_much_higher_than_stage5.jpg` to check whether stage5 is under-rewarding scenic, silhouette, landmark, or portrait compositions.
- Use `contact_06_stage5_much_higher_than_ensemble.jpg` to check whether the ensemble is over-rewarding mood, darkness, or composition at the expense of usable A-cut choices.
- Then review rejection quality with `contact_07_stage5_bottom10.jpg` and `contact_08_ensemble_bottom10.jpg`.
- Use `contact_09_bottom10_overlap.jpg` to see what both paths reject.
- Use `contact_10_bottom10_disagreements.jpg` to inspect images that only one path placed in the bottom 10.
- Use `contact_11_ensemble_much_lower_than_stage5.jpg` and `contact_12_stage5_much_lower_than_ensemble.jpg` to decide which low-ranked disagreements matter for practical A-cut rejection.

## CSV Fill-In

In `human_acut_comparison_sheet.csv`, fill:

- `human_pick`: suggested values are `stage5`, `ensemble`, `both`, `neither`, or `unsure`.
- `human_note`: short reason, for example `better landmark framing`, `too dark`, `stronger portrait`, or `duplicate`.
- `human_reject_pick`: suggested values are `stage5`, `ensemble`, `both`, `neither`, or `unsure`.
- `human_reject_note`: short reason, for example `properly rejected`, `too harsh`, `too dark`, `weak detail`, or `should not be bottom 10`.

## Rank Difference Convention

`rank_difference = ensemble_final_rank - stage5_final_rank`.

- Negative value: ensemble ranks the image higher.
- Positive value: stage5 ranks the image higher.
- Zero: same final rank.

For bottom-ranked review, the same column also tells you which model rejects an image more strongly:

- Positive value: ensemble ranks the image lower.
- Negative value: stage5 ranks the image lower.

## Summary

- Stage5 top-10 count: {len(stage5_top)}
- Ensemble top-10 count: {len(ensemble_top)}
- Top-10 overlap count: {len(overlap)}
- Stage5-only top-10: {", ".join(stage5_only["image_file_name"].tolist()) or "none"}
- Ensemble-only top-10: {", ".join(ensemble_only["image_file_name"].tolist()) or "none"}
- Stage5 bottom-10 count: {len(stage5_bottom)}
- Ensemble bottom-10 count: {len(ensemble_bottom)}
- Bottom-10 overlap count: {len(bottom_overlap)}
- Stage5-only bottom-10: {", ".join(stage5_only_bottom["image_file_name"].tolist()) or "none"}
- Ensemble-only bottom-10: {", ".join(ensemble_only_bottom["image_file_name"].tolist()) or "none"}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    html_sections = [
        ("Stage5 Top 10", sheets["stage5_top10"]),
        ("Ensemble Top 10", sheets["ensemble_top10"]),
        ("Top-10 Overlap", sheets["top10_overlap"]),
        ("Top-10 Disagreements", sheets["top10_disagreements"]),
        ("Ensemble Much Higher", sheets["ensemble_much_higher"]),
        ("Stage5 Much Higher", sheets["stage5_much_higher"]),
        ("Stage5 Bottom 10", sheets["stage5_bottom10"]),
        ("Ensemble Bottom 10", sheets["ensemble_bottom10"]),
        ("Bottom-10 Overlap", sheets["bottom10_overlap"]),
        ("Bottom-10 Disagreements", sheets["bottom10_disagreements"]),
        ("Ensemble Much Lower", sheets["ensemble_much_lower"]),
        ("Stage5 Much Lower", sheets["stage5_much_lower"]),
    ]
    html_body = "\n".join(
        f"<section><h2>{title}</h2><img src=\"{path}\" alt=\"{title}\"></section>"
        for title, path in html_sections
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage5 vs Weighted Ensemble Human A-cut Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin-bottom: 0; }}
    .meta {{ color: #555; margin-top: 4px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    section {{ margin: 28px 0; }}
    code {{ background: #f5f5f5; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Stage5 vs Weighted Ensemble Human A-cut Review</h1>
  <p class="meta">Open <code>human_acut_comparison_sheet.csv</code> to record manual picks.</p>
  <p>Top review: negative rank difference means the ensemble ranks the image higher; positive means stage5 ranks it higher.</p>
  <p>Bottom review: positive rank difference means the ensemble ranks the image lower; negative means stage5 ranks it lower.</p>
  {html_body}
</body>
</html>
"""
    (output_dir / "review_index.html").write_text(html, encoding="utf-8")


def write_package(frame: pd.DataFrame, source_csv: Path, output_dir: Path, top_k: int, sheet_limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    review = build_review_frame(frame, top_k=top_k)
    review.to_csv(output_dir / "human_acut_comparison_sheet.csv", index=False)
    review[review["in_stage5_top10"]].sort_values("stage5_final_rank").to_csv(output_dir / "stage5_top10.csv", index=False)
    review[review["in_ensemble_top10"]].sort_values("ensemble_final_rank").to_csv(output_dir / "ensemble_top10.csv", index=False)
    review[review["in_stage5_top10"] & review["in_ensemble_top10"]].sort_values("stage5_final_rank").to_csv(
        output_dir / "top10_overlap.csv",
        index=False,
    )
    review[review["disagreement_type"].isin(["stage5_only_top10", "ensemble_only_top10"])].to_csv(
        output_dir / "top10_disagreements.csv",
        index=False,
    )
    review[review["in_stage5_bottom10"]].sort_values("stage5_final_rank", ascending=False).to_csv(
        output_dir / "stage5_bottom10.csv",
        index=False,
    )
    review[review["in_ensemble_bottom10"]].sort_values("ensemble_final_rank", ascending=False).to_csv(
        output_dir / "ensemble_bottom10.csv",
        index=False,
    )
    review[review["in_stage5_bottom10"] & review["in_ensemble_bottom10"]].sort_values(
        "stage5_final_rank",
        ascending=False,
    ).to_csv(
        output_dir / "bottom10_overlap.csv",
        index=False,
    )
    review[review["bottom_disagreement_type"].isin(["stage5_only_bottom10", "ensemble_only_bottom10"])].to_csv(
        output_dir / "bottom10_disagreements.csv",
        index=False,
    )
    review.sort_values("rank_difference", ascending=True).to_csv(output_dir / "ensemble_ranks_higher.csv", index=False)
    review.sort_values("rank_difference", ascending=False).to_csv(output_dir / "stage5_ranks_higher.csv", index=False)
    review.sort_values("rank_difference", ascending=False).to_csv(output_dir / "ensemble_much_lower_than_stage5.csv", index=False)
    review.sort_values("rank_difference", ascending=True).to_csv(output_dir / "stage5_much_lower_than_ensemble.csv", index=False)
    (output_dir / "human_acut_comparison_sheet.json").write_text(
        json.dumps(safe_json_records(review), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sheets = write_contact_sheets(review, output_dir=output_dir, top_k=top_k, sheet_limit=sheet_limit)
    write_text_outputs(review, output_dir=output_dir, source_csv=source_csv, top_k=top_k, sheets=sheets)


def main() -> None:
    args = build_parser().parse_args()
    source_csv = resolve_path(args.source_csv)
    output_dir = resolve_path(args.output_dir)
    frame = load_source_frame(source_csv)
    write_package(
        frame,
        source_csv=source_csv,
        output_dir=output_dir,
        top_k=max(1, int(args.top_k)),
        sheet_limit=max(1, int(args.sheet_limit)),
    )
    print(
        json.dumps(
            {
                "source_csv": str(source_csv),
                "output_dir": str(output_dir),
                "num_images": int(len(frame)),
                "heavy_rescoring": False,
                "first_file": str(output_dir / "review_index.html"),
                "manual_csv": str(output_dir / "human_acut_comparison_sheet.csv"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
