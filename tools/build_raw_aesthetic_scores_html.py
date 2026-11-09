#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = REPO_ROOT / "outputs" / "test_vila_aesthetic_ensemble_experiment_20260423" / "scores_with_ensemble.csv"
DEFAULT_SOURCE_SUMMARY = REPO_ROOT / "outputs" / "test_vila_aesthetic_ensemble_experiment_20260423" / "summary.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "test_vila_raw_aesthetic_scores_html_20260424"

REQUIRED_COLUMNS = {
    "image_path",
    "aadb_composition",
    "nima_mean_score_raw",
    "nima_ava_unit",
    "alamp_aadb",
    "rgnet_aadb",
}

REQUIRED_MODELS = {
    "aadb_model": "AADB composition",
    "nima_model": "NIMA",
    "alamp_model": "A-Lamp",
    "rgnet_model": "RGNet",
}

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

DEFAULT_WEIGHT_NIMA_UNIT = 0.333
DEFAULT_WEIGHT_RGNET = 0.333
DEFAULT_WEIGHT_ALAMP = 0.334


def register_optional_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except Exception:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a raw aesthetic score HTML review package for test_vila using existing "
            "NIMA, RGNet, A-Lamp, and AADB composition outputs."
        )
    )
    parser.add_argument("--source_csv", default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument("--source_summary", default=str(DEFAULT_SOURCE_SUMMARY))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--disagreement_k", type=int, default=16)
    parser.add_argument("--thumb_width", type=int, default=320)
    parser.add_argument("--thumb_height", type=int, default=240)
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


def load_source_summary(source_summary: Path) -> dict[str, object]:
    if not source_summary.exists():
        raise FileNotFoundError(f"Source summary not found: {source_summary}")
    payload = json.loads(source_summary.read_text(encoding="utf-8"))
    model_paths = payload.get("model_paths")
    if not isinstance(model_paths, dict):
        raise ValueError(f"Source summary missing model_paths: {source_summary}")
    missing = sorted(key for key in REQUIRED_MODELS if key not in model_paths)
    if missing:
        raise ValueError(f"Source summary missing required model paths: {', '.join(missing)}")
    return payload


def score_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
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


def rank_desc(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(method="min", ascending=False, na_option="bottom").astype(int)


def model_min_max(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()), float(values.max())


def normalize_with_min_max(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.5
    return float((value - min_value) / (max_value - min_value))


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def color_for_level(level: float, *, low_hue: int = 5, high_hue: int = 132) -> str:
    hue = low_hue + (high_hue - low_hue) * clip01(level)
    return f"hsl({hue:.1f} 70% 90%)"


def color_for_disagreement(level: float) -> str:
    hue = 50 - 40 * clip01(level)
    return f"hsl({hue:.1f} 90% 88%)"


def load_image_rgb(image_path: Path) -> Image.Image:
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        image.load()
    return image


def save_thumbnail(
    image_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
) -> tuple[bool, str | None]:
    try:
        image = load_image_rgb(image_path)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        canvas.save(output_path, quality=90)
        return True, None
    except Exception as exc:
        canvas = Image.new("RGB", (width, height), "#f0f0f0")
        draw = ImageDraw.Draw(canvas)
        draw.multiline_text((12, 12), f"Preview unavailable\n{image_path.name}\n{exc}", fill="black", font=ImageFont.load_default(), spacing=3)
        canvas.save(output_path, quality=90)
        return False, f"{exc.__class__.__name__}: {exc}"


def sheet_label(row: pd.Series, model_label: str, score_col: str, rank_col: str) -> str:
    return (
        f"{row['image_file_name']}\n"
        f"{model_label} #{int(row[rank_col])} {float(row[score_col]):.3f}\n"
        f"NIMA {float(row['nima_score']):.3f} | RG {float(row['rgnet_score']):.3f}\n"
        f"AL {float(row['alamp_score']):.3f} | AADB {float(row['aadb_score']):.3f}"
    )


def make_contact_sheet(
    rows: pd.DataFrame,
    *,
    base_dir: Path,
    title: str,
    model_label: str,
    score_col: str,
    rank_col: str,
    output_path: Path,
    limit: int,
    thumb_width: int = 240,
    thumb_height: int = 180,
    columns: int = 5,
) -> None:
    rows = rows.head(limit).reset_index(drop=True)
    label_h, title_h, pad = 62, 36, 8
    grid_rows = max(1, math.ceil(max(1, len(rows)) / columns))
    canvas = Image.new(
        "RGB",
        (columns * thumb_width + (columns + 1) * pad, title_h + grid_rows * (thumb_height + label_h + pad) + pad),
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
        x = pad + col * (thumb_width + pad)
        y = title_h + pad + grid_row * (thumb_height + label_h + pad)
        thumb = Image.open(base_dir / row["thumbnail_path"]).convert("RGB")
        canvas.paste(thumb.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (x, y))
        draw.multiline_text((x, y + thumb_height + 4), sheet_label(row, model_label, score_col, rank_col), fill="black", font=font, spacing=2)

    canvas.save(output_path, quality=92)


def prepare_frame(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    numeric_columns = [
        "nima_mean_score_raw",
        "nima_ava_unit",
        "rgnet_aadb",
        "alamp_aadb",
        "aadb_composition",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["image_file_name"] = frame["image_path"].map(lambda value: Path(str(value)).name)
    frame["nima_score"] = frame["nima_mean_score_raw"]
    frame["nima_unit_score"] = frame["nima_ava_unit"]
    frame["rgnet_score"] = frame["rgnet_aadb"]
    frame["alamp_score"] = frame["alamp_aadb"]
    frame["aadb_score"] = frame["aadb_composition"]
    frame["review_hint_group"] = frame["image_file_name"].map(LOW_LIGHT_GROUPS).fillna("")

    frame["rank_nima"] = rank_desc(frame["nima_score"])
    frame["rank_rgnet"] = rank_desc(frame["rgnet_score"])
    frame["rank_alamp"] = rank_desc(frame["alamp_score"])
    frame["rank_aadb"] = rank_desc(frame["aadb_score"])

    unit_cols = ["nima_unit_score", "rgnet_score", "alamp_score", "aadb_score"]
    frame["unit_score_mean"] = frame[unit_cols].mean(axis=1)
    frame["unit_score_std"] = frame[unit_cols].std(axis=1, ddof=0)
    frame["unit_score_range"] = frame[unit_cols].max(axis=1) - frame[unit_cols].min(axis=1)
    frame["top_model_name"] = frame[unit_cols].idxmax(axis=1).map(
        {
            "nima_unit_score": "nima_unit",
            "rgnet_score": "rgnet",
            "alamp_score": "alamp",
            "aadb_score": "aadb",
        }
    )
    frame["bottom_model_name"] = frame[unit_cols].idxmin(axis=1).map(
        {
            "nima_unit_score": "nima_unit",
            "rgnet_score": "rgnet",
            "alamp_score": "alamp",
            "aadb_score": "aadb",
        }
    )
    frame["top_minus_bottom_unit_gap"] = frame[unit_cols].max(axis=1) - frame[unit_cols].min(axis=1)
    frame["alamp_minus_peer_mean"] = frame["alamp_score"] - frame[["nima_unit_score", "rgnet_score", "aadb_score"]].mean(axis=1)
    frame["nima_minus_peer_mean"] = frame["nima_unit_score"] - frame[["rgnet_score", "alamp_score", "aadb_score"]].mean(axis=1)
    frame["rank_disagreement"] = rank_desc(frame["unit_score_range"])

    main_columns = [
        "image_file_name",
        "image_path",
        "nima_score",
        "nima_unit_score",
        "rgnet_score",
        "alamp_score",
        "aadb_score",
        "rank_nima",
        "rank_rgnet",
        "rank_alamp",
        "rank_aadb",
        "unit_score_mean",
        "unit_score_std",
        "unit_score_range",
        "top_model_name",
        "bottom_model_name",
        "top_minus_bottom_unit_gap",
        "rank_disagreement",
        "review_hint_group",
        "alamp_minus_peer_mean",
        "nima_minus_peer_mean",
    ]
    return frame[main_columns].sort_values(["unit_score_range", "image_file_name"], ascending=[False, True]).reset_index(drop=True)


def write_csv_and_json(frame: pd.DataFrame, csv_path: Path, json_path: Path | None = None) -> None:
    frame.to_csv(csv_path, index=False)
    if json_path is not None:
        json_path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")


def build_thumbnails(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    thumb_width: int,
    thumb_height: int,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    thumbs_dir = output_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[dict[str, str]] = []
    thumb_paths: list[str] = []

    for index, row in frame.iterrows():
        source_image = resolve_repo_path(str(row["image_path"]))
        thumb_name = f"{index + 1:03d}_{row['image_file_name'].replace('/', '_')}.jpg"
        thumb_path = thumbs_dir / thumb_name
        ok, message = save_thumbnail(source_image, thumb_path, width=thumb_width, height=thumb_height)
        thumb_paths.append(str(thumb_path.relative_to(output_dir)))
        if not ok and message:
            warnings.append(
                {
                    "image_file_name": row["image_file_name"],
                    "image_path": str(row["image_path"]),
                    "thumbnail_path": str(thumb_path.relative_to(output_dir)),
                    "warning": message,
                }
            )

    enriched = frame.copy()
    enriched["thumbnail_path"] = thumb_paths
    return enriched, warnings


def model_stats_payload(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "nima_score": score_summary(frame["nima_score"]),
        "nima_unit_score": score_summary(frame["nima_unit_score"]),
        "rgnet_score": score_summary(frame["rgnet_score"]),
        "alamp_score": score_summary(frame["alamp_score"]),
        "aadb_score": score_summary(frame["aadb_score"]),
    }


def top_rows(frame: pd.DataFrame, score_col: str, rank_col: str, top_k: int) -> pd.DataFrame:
    return frame.sort_values([rank_col, "image_file_name"]).head(top_k)


def disagreement_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    return frame.sort_values(
        ["unit_score_range", "unit_score_std", "alamp_minus_peer_mean", "image_file_name"],
        ascending=[False, False, False, True],
    ).head(limit)


def write_review_index(
    output_dir: Path,
    frame: pd.DataFrame,
    *,
    model_stats: dict[str, object],
    checkpoint_paths: dict[str, str],
    top_k: int,
    disagreement_k: int,
) -> None:
    del model_stats, checkpoint_paths, top_k, disagreement_k
    model_ranges = {
        "nima_unit_score": model_min_max(frame, "nima_unit_score"),
        "rgnet_score": model_min_max(frame, "rgnet_score"),
        "alamp_score": model_min_max(frame, "alamp_score"),
    }

    table_rows = []
    for _, row in frame.iterrows():
        orig_rel = Path("..") / Path("..") / row["image_path"]
        nima_unit_level = normalize_with_min_max(float(row["nima_unit_score"]), *model_ranges["nima_unit_score"])
        rgnet_level = normalize_with_min_max(float(row["rgnet_score"]), *model_ranges["rgnet_score"])
        alamp_level = normalize_with_min_max(float(row["alamp_score"]), *model_ranges["alamp_score"])
        image_name = html.escape(row["image_file_name"])
        hint = html.escape(str(row["review_hint_group"]) or "")
        table_rows.append(
            "<tr "
            f"data-image-name='{image_name}' "
            f"data-hint='{hint}' "
            f"data-nima-unit='{float(row['nima_unit_score']):.12f}' "
            f"data-rgnet='{float(row['rgnet_score']):.12f}' "
            f"data-alamp='{float(row['alamp_score']):.12f}'"
            ">"
            f"<td><a href='{html.escape(str(orig_rel))}'><img class='thumb' src='{html.escape(row['thumbnail_path'])}' alt='{image_name}'></a></td>"
            f"<td>{image_name}</td>"
            f"<td>{hint}</td>"
            f"<td class='score' style='background:{color_for_level(nima_unit_level)}' data-sort='{float(row['nima_unit_score']):.6f}'>{float(row['nima_unit_score']):.3f}</td>"
            f"<td class='score' style='background:{color_for_level(rgnet_level)}' data-sort='{float(row['rgnet_score']):.6f}'>{float(row['rgnet_score']):.3f}</td>"
            f"<td class='score' style='background:{color_for_level(alamp_level)}' data-sort='{float(row['alamp_score']):.6f}'>{float(row['alamp_score']):.3f}</td>"
            "<td class='score combined-score' data-sort=''></td>"
            "<td class='score combined-rank' data-sort=''></td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Manual Aesthetic Weight Tuning</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 20px; color: #111; background: #fafafa; }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ line-height: 1.45; }}
    code {{ background: #f0f0f0; padding: 1px 4px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: middle; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #f4f4f4; cursor: pointer; }}
    .thumb {{ width: 120px; height: 90px; object-fit: cover; display: block; background: #fff; }}
    .table-wrap {{ overflow-x: auto; }}
    .score {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .weight-panel {{ background: white; border: 1px solid #ddd; padding: 16px; margin: 16px 0 18px; }}
    .weights {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 12px 0; }}
    .weights label {{ display: flex; flex-direction: column; gap: 6px; font-size: 13px; }}
    .weights input {{ padding: 8px; font: inherit; }}
    .status-row {{ display: flex; gap: 18px; flex-wrap: wrap; align-items: center; margin-top: 10px; font-size: 13px; }}
    .status-ok {{ color: #166534; }}
    .status-warn {{ color: #b45309; }}
    .formula {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .controls {{ margin: 0 0 14px; display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
    .controls input {{ padding: 6px 8px; min-width: 280px; }}
    .muted {{ color: #555; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Manual Aesthetic Weight Tuning</h1>
  <p>This page is a manual weight-tuning viewer for exactly three aesthetic model scores: <strong>NIMA unit</strong>, <strong>RGNet</strong>, and <strong>A-Lamp</strong>. It reuses the existing raw score package. No rescoring, technical score, production mix, or explanation pass is involved here.</p>
  <div class="weight-panel">
    <h2>Live Weights</h2>
    <p class="muted">The three weights must sum to exactly <code>1.0</code>. When the sum is valid, the table recomputes a live combined score and combined rank for every image.</p>
    <div class="weights">
      <label>
        <span>NIMA unit weight</span>
        <input id="weightNimaUnit" type="number" min="0" max="1" step="0.001" value="{DEFAULT_WEIGHT_NIMA_UNIT:.3f}">
      </label>
      <label>
        <span>RGNet weight</span>
        <input id="weightRgnet" type="number" min="0" max="1" step="0.001" value="{DEFAULT_WEIGHT_RGNET:.3f}">
      </label>
      <label>
        <span>A-Lamp weight</span>
        <input id="weightAlamp" type="number" min="0" max="1" step="0.001" value="{DEFAULT_WEIGHT_ALAMP:.3f}">
      </label>
    </div>
    <div class="status-row">
      <span>Current sum: <strong id="weightSum">1.000</strong></span>
      <span id="weightStatus" class="status-ok">Valid</span>
      <span class="formula" id="weightFormula"></span>
    </div>
  </div>
  <section>
    <h2>All images</h2>
    <div class="controls">
      <label>Filter file or hint: <input id="filterInput" type="text" placeholder="night, silhouette, IMG_2729, ..."></label>
      <span>{len(frame)} images</span>
    </div>
    <div class="table-wrap">
      <table id="scoreTable">
        <thead>
          <tr>
            <th>Preview</th>
            <th>Image</th>
            <th>Hint</th>
            <th>NIMA unit</th>
            <th>RGNet</th>
            <th>A-Lamp</th>
            <th>Combined score</th>
            <th>Combined rank</th>
          </tr>
        </thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </div>
  </section>

  <script>
    const table = document.getElementById('scoreTable');
    const headers = table.querySelectorAll('th');
    const tbody = table.querySelector('tbody');
    const filterInput = document.getElementById('filterInput');
    const weightNimaUnit = document.getElementById('weightNimaUnit');
    const weightRgnet = document.getElementById('weightRgnet');
    const weightAlamp = document.getElementById('weightAlamp');
    const weightSum = document.getElementById('weightSum');
    const weightStatus = document.getElementById('weightStatus');
    const weightFormula = document.getElementById('weightFormula');
    let sortState = {{ index: null, asc: false }};
    const combinedScoreColumn = 6;
    const combinedRankColumn = 7;

    function readWeight(input) {{
      const value = Number(input.value);
      return Number.isFinite(value) ? value : NaN;
    }}

    function updateCombinedScores() {{
      const wNima = readWeight(weightNimaUnit);
      const wRgnet = readWeight(weightRgnet);
      const wAlamp = readWeight(weightAlamp);
      const sum = wNima + wRgnet + wAlamp;
      const valid = Number.isFinite(sum) && Math.abs(sum - 1.0) <= 0.0005;

      weightSum.textContent = Number.isFinite(sum) ? sum.toFixed(3) : 'NaN';
      if (valid) {{
        weightStatus.textContent = 'Valid';
        weightStatus.className = 'status-ok';
      }} else {{
        weightStatus.textContent = 'Invalid: weights must sum to 1.0';
        weightStatus.className = 'status-warn';
      }}
      weightFormula.textContent =
        `combined = (${{wNima.toFixed(3)}} * nima_unit) + (${{wRgnet.toFixed(3)}} * rgnet) + (${{wAlamp.toFixed(3)}} * alamp)`;

      const rows = Array.from(tbody.querySelectorAll('tr'));
      const combinedRows = [];
      rows.forEach((row) => {{
        const scoreCell = row.children[combinedScoreColumn];
        const rankCell = row.children[combinedRankColumn];
        if (!valid) {{
          scoreCell.textContent = 'invalid';
          scoreCell.dataset.sort = '';
          rankCell.textContent = 'invalid';
          rankCell.dataset.sort = '';
          return;
        }}
        const nima = Number(row.dataset.nimaUnit);
        const rgnet = Number(row.dataset.rgnet);
        const alamp = Number(row.dataset.alamp);
        const combined = (wNima * nima) + (wRgnet * rgnet) + (wAlamp * alamp);
        row.dataset.combined = combined.toFixed(12);
        scoreCell.textContent = combined.toFixed(3);
        scoreCell.dataset.sort = combined.toFixed(12);
        combinedRows.push({{ row, combined }});
      }});

      if (valid) {{
        combinedRows.sort((a, b) => b.combined - a.combined);
        combinedRows.forEach((item, index) => {{
          const rank = index + 1;
          const rankCell = item.row.children[combinedRankColumn];
          rankCell.textContent = String(rank);
          rankCell.dataset.sort = String(rank);
        }});
      }}

      if (sortState.index !== null) {{
        sortByColumn(sortState.index, sortState.asc);
      }} else if (valid) {{
        sortByColumn(combinedRankColumn, true);
      }}
    }}

    function sortByColumn(index, asc) {{
      sortState = {{ index, asc }};
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {{
        const aCell = a.children[index];
        const bCell = b.children[index];
        const aSort = aCell.dataset.sort ?? aCell.textContent.trim().toLowerCase();
        const bSort = bCell.dataset.sort ?? bCell.textContent.trim().toLowerCase();
        const aNum = Number(aSort);
        const bNum = Number(bSort);
        let cmp = 0;
        if (!Number.isNaN(aNum) && !Number.isNaN(bNum) && aSort !== '' && bSort !== '') {{
          cmp = aNum - bNum;
        }} else {{
          cmp = String(aSort).localeCompare(String(bSort));
        }}
        return asc ? cmp : -cmp;
      }});
      rows.forEach((row) => tbody.appendChild(row));
    }}

    headers.forEach((header, index) => {{
      header.addEventListener('click', () => {{
        const asc = sortState.index === index ? !sortState.asc : true;
        sortByColumn(index, asc);
      }});
    }});

    filterInput.addEventListener('input', () => {{
      const needle = filterInput.value.trim().toLowerCase();
      Array.from(tbody.querySelectorAll('tr')).forEach((row) => {{
        const text = row.textContent.toLowerCase();
        row.style.display = !needle || text.includes(needle) ? '' : 'none';
      }});
    }});

    [weightNimaUnit, weightRgnet, weightAlamp].forEach((input) => {{
      input.addEventListener('input', updateCombinedScores);
    }});

    updateCombinedScores();
  </script>
</body>
</html>
"""
    (output_dir / "review_index.html").write_text(page, encoding="utf-8")


def build_outputs(
    frame: pd.DataFrame,
    *,
    source_csv: Path,
    source_summary: Path,
    output_dir: Path,
    source_summary_payload: dict[str, object],
    top_k: int,
    disagreement_k: int,
    thumb_width: int,
    thumb_height: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, thumb_warnings = build_thumbnails(
        frame,
        output_dir=output_dir,
        thumb_width=thumb_width,
        thumb_height=thumb_height,
    )

    main_csv = output_dir / "raw_aesthetic_scores.csv"
    write_csv_and_json(frame, main_csv)

    ranked_specs = [
        ("ranked_nima.csv", "rank_nima", "nima_score"),
        ("ranked_rgnet.csv", "rank_rgnet", "rgnet_score"),
        ("ranked_alamp.csv", "rank_alamp", "alamp_score"),
        ("ranked_aadb.csv", "rank_aadb", "aadb_score"),
    ]
    for file_name, rank_col, score_col in ranked_specs:
        ranked = frame.sort_values([rank_col, "image_file_name"])
        write_csv_and_json(ranked, output_dir / file_name)

    write_csv_and_json(pd.DataFrame(thumb_warnings), output_dir / "thumbnail_warnings.csv")

    top_nima = top_rows(frame, "nima_score", "rank_nima", top_k)
    top_rgnet = top_rows(frame, "rgnet_score", "rank_rgnet", top_k)
    top_alamp = top_rows(frame, "alamp_score", "rank_alamp", top_k)
    top_aadb = top_rows(frame, "aadb_score", "rank_aadb", top_k)
    high_disagreement = disagreement_rows(frame, disagreement_k)

    make_contact_sheet(
        top_nima,
        base_dir=output_dir,
        title="Top 10 NIMA",
        model_label="NIMA",
        score_col="nima_score",
        rank_col="rank_nima",
        output_path=output_dir / "contact_top10_nima.jpg",
        limit=top_k,
    )
    make_contact_sheet(
        top_rgnet,
        base_dir=output_dir,
        title="Top 10 RGNet",
        model_label="RGNet",
        score_col="rgnet_score",
        rank_col="rank_rgnet",
        output_path=output_dir / "contact_top10_rgnet.jpg",
        limit=top_k,
    )
    make_contact_sheet(
        top_alamp,
        base_dir=output_dir,
        title="Top 10 A-Lamp",
        model_label="A-Lamp",
        score_col="alamp_score",
        rank_col="rank_alamp",
        output_path=output_dir / "contact_top10_alamp.jpg",
        limit=top_k,
    )
    make_contact_sheet(
        top_aadb,
        base_dir=output_dir,
        title="Top 10 AADB composition",
        model_label="AADB",
        score_col="aadb_score",
        rank_col="rank_aadb",
        output_path=output_dir / "contact_top10_aadb.jpg",
        limit=top_k,
    )
    make_contact_sheet(
        high_disagreement,
        base_dir=output_dir,
        title="High raw-score disagreement",
        model_label="Range",
        score_col="unit_score_range",
        rank_col="rank_disagreement",
        output_path=output_dir / "contact_high_disagreement.jpg",
        limit=min(disagreement_k, len(high_disagreement)),
    )

    dataset_images = sorted(resolve_repo_path("test_vila").glob("*"))
    dataset_image_files = [path for path in dataset_images if path.is_file()]
    required_model_paths = {key: str(source_summary_payload["model_paths"][key]) for key in REQUIRED_MODELS}
    model_stats = model_stats_payload(frame)

    summary = {
        "experiment_name": "raw_aesthetic_scores_html_review",
        "non_production": True,
        "input_dir": str(REPO_ROOT / "test_vila"),
        "source_csv": str(source_csv),
        "source_summary": str(source_summary),
        "output_dir": str(output_dir),
        "num_dataset_images": int(len(dataset_image_files)),
        "num_scored_rows": int(len(frame)),
        "scoring_source_reused": True,
        "scoring_scripts": [
            str(REPO_ROOT / "tools" / "run_aesthetic_ensemble_experiment.py"),
            str(REPO_ROOT / "src" / "infer" / "score_image_folder.py"),
            str(REPO_ROOT / "src" / "infer" / "predict_quality_bundle.py"),
        ],
        "review_builder_script": str(Path(__file__).resolve()),
        "checkpoint_paths": required_model_paths,
        "model_names": ["nima", "rgnet", "alamp", "aadb_composition"],
        "score_summary": model_stats,
        "all_four_scores_present_for_all_rows": bool(
            frame[["nima_score", "nima_unit_score", "rgnet_score", "alamp_score", "aadb_score"]].notna().all().all()
        ),
        "skipped_images": [],
        "thumbnail_warnings": thumb_warnings,
        "top10_by_model": {
            "nima": top_nima["image_file_name"].tolist(),
            "rgnet": top_rgnet["image_file_name"].tolist(),
            "alamp": top_alamp["image_file_name"].tolist(),
            "aadb": top_aadb["image_file_name"].tolist(),
        },
        "high_disagreement_images": high_disagreement["image_file_name"].tolist(),
        "artifacts": [],
    }

    write_review_index(
        output_dir,
        frame,
        model_stats=model_stats,
        checkpoint_paths=required_model_paths,
        top_k=top_k,
        disagreement_k=disagreement_k,
    )
    summary["artifacts"] = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    register_optional_heif()
    args = build_parser().parse_args()
    source_csv = resolve_repo_path(args.source_csv)
    source_summary = resolve_repo_path(args.source_summary)
    output_dir = resolve_repo_path(args.output_dir)

    source_frame = load_source_frame(source_csv)
    source_summary_payload = load_source_summary(source_summary)
    frame = prepare_frame(source_frame)
    summary = build_outputs(
        frame,
        source_csv=source_csv,
        source_summary=source_summary,
        output_dir=output_dir,
        source_summary_payload=source_summary_payload,
        top_k=args.top_k,
        disagreement_k=args.disagreement_k,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
    )
    print(json.dumps({"output_dir": summary["output_dir"], "num_rows": summary["num_scored_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
