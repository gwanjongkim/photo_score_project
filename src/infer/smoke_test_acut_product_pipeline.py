from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.infer.acut_product_export import (
    APP_DEBUG_FIELDS,
    APP_RESULT_FIELDS,
    APP_RESULTS_SCHEMA_VERSION,
    SUMMARY_REQUIRED_FIELDS,
    TOP_K_SUMMARY_ITEM_FIELDS,
)


FALLBACK_SCORE_PATHS = [
    "outputs/acut_stage5_full_with_pairwise/scores.csv",
    "outputs/stage5_runs/stage5_smoke_test/scores.csv",
    "outputs/acut_full_stack/scores.csv",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the productized A-cut pipeline exports.")
    parser.add_argument("--scores_csv", default="outputs/acut_stage5_full_with_pairwise/scores.csv")
    parser.add_argument("--vila_scores_csv", default="outputs/vila_scores_run/vila_scores.csv")
    parser.add_argument("--output_dir", default="outputs/acut_product_ready_smoke")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--enable_diversity", action="store_true")
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def resolve_scores_csv(repo_root: Path, requested_path: str) -> tuple[Path, str | None]:
    requested = repo_root / requested_path
    if requested.exists():
        return requested, None

    for candidate in FALLBACK_SCORE_PATHS:
        candidate_path = repo_root / candidate
        if candidate_path.exists():
            return candidate_path, f"Requested scores CSV missing at {requested_path}; using {candidate} instead."

    raise FileNotFoundError(
        f"Could not find {requested_path} and no fallback score CSVs were available under outputs/."
    )


def run_pipeline(
    repo_root: Path,
    *,
    scores_csv: Path,
    vila_scores_csv: Path,
    output_dir: Path,
    top_k: int,
    enable_diversity: bool,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "src/infer/run_acut_pipeline.py",
        "--scores_csv",
        str(scores_csv),
        "--vila_scores_csv",
        str(vila_scores_csv),
        "--output_dir",
        str(output_dir),
        "--top_k",
        str(top_k),
    ]
    if enable_diversity:
        command.append("--enable_diversity")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"
    return subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def verify_outputs(output_dir: Path, top_k: int, *, expect_diversity: bool) -> dict[str, object]:
    app_results_json = output_dir / "app_results.json"
    top_k_summary_json = output_dir / "top_k_summary.json"
    app_results_csv = output_dir / "app_results.csv"
    review_sheet_csv = output_dir / "review_sheet.csv"

    missing_paths = [
        str(path.relative_to(output_dir.parent))
        for path in [app_results_json, app_results_csv, top_k_summary_json, review_sheet_csv]
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing expected output files: {', '.join(missing_paths)}")

    with app_results_json.open("r", encoding="utf-8") as f:
        app_rows = json.load(f)
    if not isinstance(app_rows, list) or not app_rows:
        raise ValueError("app_results.json must contain a non-empty JSON array.")

    missing_fields = [field for field in APP_RESULT_FIELDS if field not in app_rows[0]]
    if missing_fields:
        raise ValueError(f"app_results.json is missing required fields: {', '.join(missing_fields)}")
    unexpected_debug_fields = [field for field in APP_DEBUG_FIELDS if field in app_rows[0]]
    if unexpected_debug_fields:
        raise ValueError(
            "app_results.json unexpectedly included debug-only fields in the default export: "
            + ", ".join(unexpected_debug_fields)
        )

    selected_rows = [row for row in app_rows if row.get("selected")]
    rejected_rows = [row for row in app_rows if not row.get("selected")]
    if len(selected_rows) != min(int(top_k), len(app_rows)):
        raise ValueError("Selected row count in app_results.json does not match the requested top_k.")
    if not rejected_rows:
        raise ValueError("No rejected rows were found in app_results.json.")

    with top_k_summary_json.open("r", encoding="utf-8") as f:
        top_k_summary = json.load(f)

    for key in SUMMARY_REQUIRED_FIELDS:
        if key not in top_k_summary:
            raise ValueError(f"top_k_summary.json is missing required key: {key}")
    if top_k_summary["schema_version"] != APP_RESULTS_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected schema_version {top_k_summary['schema_version']!r}; expected {APP_RESULTS_SCHEMA_VERSION!r}."
        )
    if not isinstance(top_k_summary["generated_at"], str) or not top_k_summary["generated_at"].strip():
        raise ValueError("generated_at in top_k_summary.json must be a non-empty string.")
    if not isinstance(top_k_summary["score_semantics"], str) or not top_k_summary["score_semantics"].strip():
        raise ValueError("score_semantics in top_k_summary.json must be a non-empty string.")

    if not isinstance(top_k_summary["top_k"], list) or not top_k_summary["top_k"]:
        raise ValueError("top_k_summary.json must contain a non-empty top_k array.")

    top_k_item_missing_fields = [
        field for field in TOP_K_SUMMARY_ITEM_FIELDS if field not in top_k_summary["top_k"][0]
    ]
    if top_k_item_missing_fields:
        raise ValueError(
            "top_k_summary.json top_k items are missing required fields: "
            + ", ".join(top_k_item_missing_fields)
        )

    if int(top_k_summary["selected_count"]) != len(selected_rows):
        raise ValueError("selected_count in top_k_summary.json does not match app_results.json.")
    if int(top_k_summary["rejected_count"]) != len(rejected_rows):
        raise ValueError("rejected_count in top_k_summary.json does not match app_results.json.")
    diversity_enabled = bool(top_k_summary["diversity_enabled"])
    if diversity_enabled != expect_diversity:
        raise ValueError(
            f"diversity_enabled in top_k_summary.json was {diversity_enabled}, expected {expect_diversity}."
        )
    expected_ranking_stage = "post_diversity" if expect_diversity else "post_rerank"
    if top_k_summary["ranking_stage"] != expected_ranking_stage:
        raise ValueError(
            f"ranking_stage in top_k_summary.json was {top_k_summary['ranking_stage']!r}, "
            f"expected {expected_ranking_stage!r}."
        )
    score_semantics = top_k_summary["score_semantics"]
    if expect_diversity and "pre-diversity" not in score_semantics:
        raise ValueError("score_semantics must mention pre-diversity behavior when diversity is enabled.")
    if not expect_diversity and "diversity reranking is disabled" not in score_semantics:
        raise ValueError("score_semantics must explain the non-diversity path when diversity is disabled.")

    return {
        "rows": len(app_rows),
        "selected_rows": len(selected_rows),
        "rejected_rows": len(rejected_rows),
        "selected_example": selected_rows[0],
        "rejected_example": rejected_rows[0],
        "app_result_item_keys": sorted(app_rows[0].keys()),
        "top_k_summary_keys": sorted(top_k_summary.keys()),
        "schema_version": top_k_summary["schema_version"],
        "generated_at": top_k_summary["generated_at"],
        "ranking_stage": top_k_summary["ranking_stage"],
        "score_semantics": top_k_summary["score_semantics"],
        "diversity_enabled": top_k_summary["diversity_enabled"],
        "final_ordering_uses_diversity": top_k_summary["final_ordering_uses_diversity"],
        "final_score_matches_final_ranking": top_k_summary["final_score_matches_final_ranking"],
    }


def print_example(title: str, row: dict[str, object]) -> None:
    print(title)
    print(f"- rank={row['rank']} image={row['image_path']} status={row['status']}")
    print(f"  short: {row['acut_short_reason']}")
    print(f"  detailed: {row['acut_detailed_reason']}")
    if row.get("acut_comparison_reason"):
        print(f"  comparison: {row['acut_comparison_reason']}")


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    scores_csv, fallback_note = resolve_scores_csv(repo_root, args.scores_csv)
    vila_scores_csv = repo_root / args.vila_scores_csv
    if not vila_scores_csv.exists():
        raise FileNotFoundError(f"VILA scores CSV not found at {args.vila_scores_csv}")

    output_dir = repo_root / args.output_dir
    result = run_pipeline(
        repo_root,
        scores_csv=scores_csv,
        vila_scores_csv=vila_scores_csv,
        output_dir=output_dir,
        top_k=args.top_k,
        enable_diversity=args.enable_diversity,
    )
    verification = verify_outputs(output_dir=output_dir, top_k=args.top_k, expect_diversity=args.enable_diversity)

    summary = {
        "requested_scores_csv": args.scores_csv,
        "resolved_scores_csv": str(scores_csv.relative_to(repo_root)),
        "vila_scores_csv": args.vila_scores_csv,
        "output_dir": str(output_dir.relative_to(repo_root)),
        "pipeline_stdout": result.stdout.strip(),
        "fallback_note": fallback_note,
        "verification": {
            key: value
            for key, value in verification.items()
            if key not in {"selected_example", "rejected_example"}
        },
    }
    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))
    print("Contract keys")
    print(f"- app_results_item_keys={verification['app_result_item_keys']}")
    print(f"- top_k_summary_keys={verification['top_k_summary_keys']}")
    print("Contract metadata")
    print(f"- schema_version={verification['schema_version']}")
    print(f"- generated_at={verification['generated_at']}")
    print(f"- ranking_stage={verification['ranking_stage']}")
    print(f"- score_semantics={verification['score_semantics']}")
    print(f"- diversity_enabled={verification['diversity_enabled']}")
    print(f"- final_ordering_uses_diversity={verification['final_ordering_uses_diversity']}")
    print(
        f"- final_score_matches_final_ranking={verification['final_score_matches_final_ranking']}"
    )
    print_example("Selected example", verification["selected_example"])
    print_example("Rejected example", verification["rejected_example"])


if __name__ == "__main__":
    main()
