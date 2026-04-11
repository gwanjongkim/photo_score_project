from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.infer.stage5_reference import ensure_project_path
from src.infer.stage5_review import (
    RUN_COMPARISON_FIELDS,
    build_run_summary,
    flatten_product_row,
    format_float,
    load_duplicate_report,
    load_ranked_rows,
    load_run_manifest,
    write_csv_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple stage5 run directories with ranking-first summaries.")
    parser.add_argument("run_dirs", nargs="+", help="Stage5 run directories to compare.")
    parser.add_argument(
        "--output_dir",
        help="Destination for the comparison summary. Defaults to outputs/stage5_comparisons/<timestamp>.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / "stage5_comparisons" / f"comparison_{timestamp}"


def ensure_clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def build_comparison_markdown(
    summary_rows: list[dict[str, object]],
    per_run_top_rows: list[tuple[str, list[dict[str, object]]]],
    top_k: int,
) -> str:
    lines = [
        "# Stage5 Run Comparison",
        "",
        f"- Compared runs: `{len(summary_rows)}`",
        f"- Top-k focus: `{top_k}`",
        "",
        "| run | images | top1 | top-k signature | duplicate groups | top-k penalized | top-k pairwise adjusted |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_label") or ""),
                    str(row.get("num_images") or ""),
                    str(row.get("top1_image") or ""),
                    str(row.get("top_k_signature") or ""),
                    str(row.get("num_duplicate_groups") or 0),
                    str(row.get("top_k_penalized_images") or 0),
                    str(row.get("top_k_pairwise_adjusted_images") or 0),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Per-Run Top-K", ""])
    for run_label, rows in per_run_top_rows:
        lines.append(f"### {run_label}")
        lines.append("")
        for row in rows:
            lines.append(
                f"{row.get('rank')}. {row.get('image_name')} | "
                f"final={format_float(row.get('final_score'))} | "
                f"pairwise_delta={format_float(row.get('pairwise_rerank_delta'))} | "
                f"diversity={format_float(row.get('diversity_penalty'))}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    resolved_output_dir = ensure_project_path(args.output_dir) if args.output_dir else default_output_dir()
    if resolved_output_dir is None:
        raise ValueError("Could not resolve comparison output directory.")
    ensure_clean_output_dir(resolved_output_dir)

    comparison_rows = []
    per_run_top_rows = []
    for run_dir_arg in args.run_dirs:
        run_dir = ensure_project_path(run_dir_arg)
        if run_dir is None or not run_dir.is_dir():
            raise ValueError(f"Run directory does not exist: {run_dir_arg}")
        rows = load_ranked_rows(run_dir)
        duplicate_report = load_duplicate_report(run_dir)
        manifest = load_run_manifest(run_dir)
        summary = build_run_summary(
            output_dir=run_dir,
            rows=rows,
            duplicate_report=duplicate_report,
            manifest=manifest,
            top_k=args.top_k,
        )
        comparison_rows.append(summary)
        per_run_top_rows.append(
            (
                str(summary.get("run_label") or Path(run_dir).name),
                [flatten_product_row(row) for row in rows[: args.top_k]],
            )
        )

    summary_csv = resolved_output_dir / "comparison_summary.csv"
    summary_json = resolved_output_dir / "comparison_summary.json"
    review_md = resolved_output_dir / "comparison_review.md"

    write_csv_rows(summary_csv, RUN_COMPARISON_FIELDS, comparison_rows)
    summary_json.write_text(json.dumps(comparison_rows, indent=args.json_indent, ensure_ascii=False) + "\n", encoding="utf-8")
    review_md.write_text(
        build_comparison_markdown(comparison_rows, per_run_top_rows, top_k=args.top_k),
        encoding="utf-8",
    )

    result = {
        "output_dir": str(resolved_output_dir),
        "comparison_summary_csv": str(summary_csv),
        "comparison_summary_json": str(summary_json),
        "comparison_review_md": str(review_md),
        "num_runs": len(comparison_rows),
        "top_k": args.top_k,
    }
    print(json.dumps(result, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
