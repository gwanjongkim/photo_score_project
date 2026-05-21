# 브라우저에서 가중치를 조정할 수 있는 오프라인 HTML 리포트를 생성합니다.
from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from model_registry import ModelSpec


def safe_asset_name(index: int, source_path: Path, *, suffix: str | None = None) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.stem).strip("._-") or "image"
    ext = suffix if suffix is not None else source_path.suffix.lower()
    return f"{index:04d}_{stem}{ext}"


def make_thumbnail(source_path: Path, output_path: Path, *, size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source_path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (size, size), "white")
            canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
            canvas.save(output_path, quality=90)
    except Exception as exc:
        canvas = Image.new("RGB", (size, size), "#f3f4f6")
        draw = ImageDraw.Draw(canvas)
        draw.multiline_text(
            (12, 12),
            f"Preview unavailable\n{source_path.name}\n{exc.__class__.__name__}",
            fill="#111827",
            font=ImageFont.load_default(),
            spacing=3,
        )
        canvas.save(output_path, quality=90)


def prepare_report_assets(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    thumbnail_size: int,
    copy_original_images: bool,
) -> None:
    thumbs_dir = output_dir / "thumbs"
    copied_dir = output_dir / "copied_images"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    if copy_original_images:
        copied_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records, start=1):
        source_path = Path(str(record["image_path"]))
        thumb_name = safe_asset_name(index, source_path, suffix=".jpg")
        thumb_path = thumbs_dir / thumb_name
        make_thumbnail(source_path, thumb_path, size=thumbnail_size)
        record["thumbnail_path"] = str(Path("thumbs") / thumb_name)

        if copy_original_images:
            copied_name = safe_asset_name(index, source_path)
            copied_path = copied_dir / copied_name
            shutil.copy2(source_path, copied_path)
            record["copied_image_path"] = str(Path("copied_images") / copied_name)
        else:
            record["copied_image_path"] = None


def fmt_score(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def score_attr(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.12f}"
    except (TypeError, ValueError):
        return ""


def preset_label(preset_name: str, specs: list[ModelSpec]) -> str:
    if preset_name == "equal":
        return "Equal weights"
    if preset_name == "current_config":
        return "Current config weights"
    if preset_name.endswith("_heavy"):
        model_id = preset_name[: -len("_heavy")]
        for spec in specs:
            if spec.model_id == model_id:
                return f"{spec.display_name}-heavy"
    return preset_name.replace("_", " ").title()


def render_weight_controls(
    *,
    specs: list[ModelSpec],
    weights: dict[str, float],
    presets: dict[str, dict[str, float]],
    enable_interactive_weight_controls: bool,
) -> str:
    if not enable_interactive_weight_controls:
        return ""
    controls = []
    for spec in specs:
        value = float(weights.get(spec.model_id, 0.0))
        controls.append(
            "<label class='weight-control'>"
            f"<span>{html.escape(spec.display_name)} weight</span>"
            "<div class='weight-input-row'>"
            f"<input type='range' id='weight-slider-{html.escape(spec.model_id)}' "
            f"data-model='{html.escape(spec.model_id)}' min='0' max='1' step='0.01' value='{value:.4f}'>"
            f"<input type='number' id='weight-{html.escape(spec.model_id)}' "
            f"data-model='{html.escape(spec.model_id)}' min='0' max='1' step='0.01' value='{value:.4f}'>"
            "</div>"
            "</label>"
        )
    preset_buttons = [
        f"<button type='button' class='preset-button' data-preset='{html.escape(name)}'>{html.escape(preset_label(name, specs))}</button>"
        for name in presets
    ]
    return (
        "<section class='controls'>"
        "<div class='weight-grid'>"
        + "\n".join(controls)
        + "</div>"
        "<div class='preset-row'>"
        + "\n".join(preset_buttons)
        + "</div>"
        "<p id='weight-message' class='weight-message'></p>"
        "</section>"
    )


def render_rows(records: list[dict[str, Any]], specs: list[ModelSpec]) -> str:
    rows = []
    for record in records:
        data_attrs = [f"data-score-{html.escape(spec.model_id)}='{score_attr(record.get(spec.score_column))}'" for spec in specs]
        image_name = html.escape(str(record.get("image_name") or Path(str(record["image_path"])).name))
        thumbnail = html.escape(str(record.get("thumbnail_path") or ""))
        copied = record.get("copied_image_path")
        image_cell = image_name
        if copied:
            image_cell = f"<a href='{html.escape(str(copied))}'>{image_name}</a>"
        score_cells = []
        for spec in specs:
            score_cells.append(
                f"<td class='score-cell model-score' data-model='{html.escape(spec.model_id)}'>"
                f"{html.escape(fmt_score(record.get(spec.score_column)))}"
                "</td>"
            )
        rows.append(
            "<tr "
            + " ".join(data_attrs)
            + f" data-final='{score_attr(record.get('final_score'))}'>"
            f"<td class='rank-cell'>{html.escape(str(record.get('rank') or ''))}</td>"
            f"<td class='thumb-cell'><a href='{html.escape(str(copied or thumbnail))}'><img src='{thumbnail}' alt=''></a></td>"
            f"<td class='name-cell'>{image_cell}</td>"
            + "".join(score_cells)
            + f"<td class='score-cell final-score'>{html.escape(fmt_score(record.get('final_score')))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_report_html(
    *,
    records: list[dict[str, Any]],
    specs: list[ModelSpec],
    weights: dict[str, float],
    presets: dict[str, dict[str, float]],
    report_config: dict[str, Any],
    config_path: Path,
    output_path: Path,
) -> None:
    title = str(report_config.get("title") or "A-cut Aesthetic Weight Lab")
    enable_controls = bool(report_config.get("enable_interactive_weight_controls", True))
    model_headers = "\n".join(f"<th>{html.escape(spec.display_name)} score</th>" for spec in specs)
    rows = render_rows(records, specs)
    controls = render_weight_controls(
        specs=specs,
        weights=weights,
        presets=presets,
        enable_interactive_weight_controls=enable_controls,
    )
    payload_json = json.dumps(
        {
            "models": [
                {"id": spec.model_id, "label": spec.display_name, "score_column": spec.score_column}
                for spec in specs
            ],
            "presets": presets,
            "weights": weights,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).replace("</", "<\\/")
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #111827;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: end;
      margin-bottom: 18px;
    }}
    h1 {{
      font-size: 1.55rem;
      margin: 0 0 4px;
      letter-spacing: 0;
    }}
    .meta {{
      margin: 0;
      color: #4b5563;
      font-size: 0.92rem;
    }}
    .controls {{
      border: 1px solid #d1d5db;
      background: #ffffff;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 18px;
    }}
    .weight-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .weight-control {{
      display: grid;
      gap: 6px;
      font-size: 0.88rem;
      color: #374151;
    }}
    .weight-input-row {{
      display: grid;
      grid-template-columns: minmax(120px, 1fr) 86px;
      gap: 8px;
      align-items: center;
    }}
    .weight-control input[type="number"] {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 8px 9px;
      font: inherit;
      background: #fff;
    }}
    .weight-control input[type="range"] {{
      width: 100%;
    }}
    .preset-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    button {{
      border: 1px solid #9ca3af;
      border-radius: 6px;
      background: #f9fafb;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{
      background: #eef2ff;
      border-color: #6366f1;
    }}
    .weight-message {{
      margin: 10px 0 0;
      min-height: 1.2em;
      color: #92400e;
      font-size: 0.9rem;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      background: #ffffff;
    }}
    table {{
      width: 100%;
      min-width: 860px;
      border-collapse: collapse;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 10px;
      text-align: left;
      vertical-align: middle;
      font-size: 0.92rem;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f3f4f6;
      z-index: 1;
      color: #374151;
      font-weight: 650;
    }}
    tbody tr:hover {{
      background: #f9fafb;
    }}
    .rank-cell {{
      width: 64px;
      font-weight: 700;
      color: #111827;
    }}
    .thumb-cell {{
      width: 112px;
    }}
    .thumb-cell img {{
      width: 96px;
      height: 96px;
      object-fit: contain;
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      display: block;
    }}
    .name-cell {{
      min-width: 220px;
      word-break: break-word;
    }}
    .score-cell {{
      min-width: 110px;
      font-variant-numeric: tabular-nums;
      font-weight: 650;
    }}
    a {{
      color: #1d4ed8;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>{html.escape(title)}</h1>
      <p class="meta">{len(records)} image(s) scored from config {html.escape(str(config_path))}</p>
    </div>
  </header>
  {controls}
  <div class="table-wrap">
    <table id="score-table">
      <thead>
        <tr>
          <th>Rank</th>
          <th>Thumbnail</th>
          <th>Image name</th>
          {model_headers}
          <th>Final weighted score</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
</main>
<script id="lab-data" type="application/json">{payload_json}</script>
<script>
const payload = JSON.parse(document.getElementById('lab-data').textContent);
const models = payload.models;
const presets = payload.presets;

function clamp01(value) {{
  return Math.max(0, Math.min(1, value));
}}

function scoreColor(value) {{
  if (!Number.isFinite(value)) return '#f3f4f6';
  const hue = 5 + (130 * clamp01(value));
  return `hsl(${{hue.toFixed(1)}} 72% 88%)`;
}}

function readWeights() {{
  const weights = {{}};
  for (const model of models) {{
    const input = document.getElementById(`weight-${{model.id}}`);
    weights[model.id] = Math.max(0, Number(input?.value || 0));
  }}
  return weights;
}}

function setWeights(weights) {{
  for (const model of models) {{
    const input = document.getElementById(`weight-${{model.id}}`);
    const slider = document.getElementById(`weight-slider-${{model.id}}`);
    if (input && Object.prototype.hasOwnProperty.call(weights, model.id)) {{
      const value = clamp01(Number(weights[model.id]));
      input.value = value.toFixed(4);
      if (slider) slider.value = value.toFixed(4);
    }}
  }}
  updateScores();
}}

function syncWeightControl(modelId, source) {{
  const input = document.getElementById(`weight-${{modelId}}`);
  const slider = document.getElementById(`weight-slider-${{modelId}}`);
  if (!input || !slider) return;
  const sourceValue = source === 'slider' ? slider.value : input.value;
  const value = clamp01(Number(sourceValue || 0));
  input.value = value.toFixed(4);
  slider.value = value.toFixed(4);
}}

function updateCellColors(row) {{
  for (const model of models) {{
    const value = Number(row.getAttribute(`data-score-${{model.id}}`));
    const cell = row.querySelector(`.model-score[data-model="${{model.id}}"]`);
    if (cell) cell.style.background = scoreColor(value);
  }}
  const finalValue = Number(row.dataset.final);
  const finalCell = row.querySelector('.final-score');
  if (finalCell) finalCell.style.background = scoreColor(finalValue);
}}

function updateScores() {{
  const weights = readWeights();
  const rawWeightSum = Object.values(weights).reduce((sum, value) => sum + value, 0);
  const message = document.getElementById('weight-message');
  if (message) {{
    if (rawWeightSum <= 0) {{
      message.textContent = 'Set at least one weight above zero.';
    }} else if (Math.abs(rawWeightSum - 1) > 0.001) {{
      message.textContent = `Weights are auto-normalized from sum ${{rawWeightSum.toFixed(4)}}.`;
    }} else {{
      message.textContent = 'Weights sum to 1.0000.';
    }}
  }}

  const tbody = document.querySelector('#score-table tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  for (const row of rows) {{
    let weighted = 0;
    let activeWeight = 0;
    for (const model of models) {{
      const score = Number(row.getAttribute(`data-score-${{model.id}}`));
      const weight = weights[model.id] || 0;
      if (Number.isFinite(score) && weight > 0) {{
        weighted += weight * clamp01(score);
        activeWeight += weight;
      }}
    }}
    const finalScore = activeWeight > 0 ? weighted / activeWeight : NaN;
    row.dataset.final = Number.isFinite(finalScore) ? finalScore.toFixed(12) : '';
    const finalCell = row.querySelector('.final-score');
    if (finalCell) finalCell.textContent = Number.isFinite(finalScore) ? finalScore.toFixed(4) : '';
    updateCellColors(row);
  }}

  rows.sort((left, right) => {{
    const leftScore = Number(left.dataset.final);
    const rightScore = Number(right.dataset.final);
    if (!Number.isFinite(leftScore) && !Number.isFinite(rightScore)) return 0;
    if (!Number.isFinite(leftScore)) return 1;
    if (!Number.isFinite(rightScore)) return -1;
    return rightScore - leftScore;
  }});
  rows.forEach((row, index) => {{
    row.querySelector('.rank-cell').textContent = String(index + 1);
    tbody.appendChild(row);
  }});
}}

for (const model of models) {{
  const input = document.getElementById(`weight-${{model.id}}`);
  const slider = document.getElementById(`weight-slider-${{model.id}}`);
  if (input) input.addEventListener('input', () => {{
    syncWeightControl(model.id, 'number');
    updateScores();
  }});
  if (slider) slider.addEventListener('input', () => {{
    syncWeightControl(model.id, 'slider');
    updateScores();
  }});
}}

for (const button of document.querySelectorAll('.preset-button')) {{
  button.addEventListener('click', () => {{
    const preset = presets[button.dataset.preset];
    if (preset) setWeights(preset);
  }});
}}

updateScores();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
