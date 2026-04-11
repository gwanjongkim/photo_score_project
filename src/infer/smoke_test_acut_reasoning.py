from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPECTED_FIELDS = [
    "acut_short_reason",
    "acut_detailed_reason",
    "acut_comparison_reason",
    "acut_rejection_reason",
    "acut_explanation_structured",
]

FALLBACK_SCORE_PATHS = [
    "outputs/acut_stage5_full_with_pairwise/scores.csv",
    "outputs/stage5_runs/stage5_smoke_test/scores.csv",
    "outputs/acut_full_stack/scores.csv",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the selector-side A-cut reasoning outputs.")
    parser.add_argument("--scores_csv", default="outputs/stage5_scores/scores.csv")
    parser.add_argument("--vila_scores_csv", default="outputs/vila_scores_run/vila_scores.csv")
    parser.add_argument("--output_dir", default="outputs/stage5_with_vila_reasoning_smoke")
    parser.add_argument("--top_k", type=int, default=5)
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


def run_selector(
    repo_root: Path,
    scores_csv: Path,
    vila_scores_csv: Path,
    output_dir: Path,
    top_k: int,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "src/infer/select_best_shots.py",
        "--scores_csv",
        str(scores_csv),
        "--vila_scores_csv",
        str(vila_scores_csv),
        "--enable_vila_rerank",
        "--vila_rerank_weight",
        "0.10",
        "--enable_vila_explanations",
        "--enable_acut_reasoning",
        "--reason_reference_mode",
        "nearest_competitor",
        "--reason_detail_level",
        "full",
        "--reason_include_model_contributions",
        "--reason_include_vila_signals",
        "--reason_include_comparison",
        "--output_dir",
        str(output_dir),
        "--top_k",
        str(top_k),
    ]
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


def verify_outputs(ranked_csv: Path, top_k: int) -> dict[str, object]:
    df = pd.read_csv(ranked_csv)
    missing_fields = [field for field in EXPECTED_FIELDS if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Missing expected reasoning fields in {ranked_csv}: {', '.join(missing_fields)}")

    if df["acut_short_reason"].isna().all():
        raise ValueError("acut_short_reason is empty for every row.")
    if df["acut_detailed_reason"].isna().all():
        raise ValueError("acut_detailed_reason is empty for every row.")

    selected_df = df[df["rank"] <= top_k].copy()
    rejected_df = df[df["rank"] > top_k].copy()
    if selected_df.empty:
        raise ValueError("No selected rows were found in the ranked output.")
    if rejected_df.empty:
        raise ValueError("No rejected rows were found in the ranked output.")

    structured_nonempty = df["acut_explanation_structured"].fillna("").str.len().gt(2).sum()
    if structured_nonempty == 0:
        raise ValueError("acut_explanation_structured is empty for every row.")

    return {
        "rows": int(len(df)),
        "selected_rows": int(len(selected_df)),
        "rejected_rows": int(len(rejected_df)),
        "structured_nonempty_rows": int(structured_nonempty),
        "selected_examples": selected_df.head(3).to_dict(orient="records"),
        "rejected_examples": rejected_df.head(3).to_dict(orient="records"),
    }


def print_examples(title: str, rows: list[dict[str, object]]) -> None:
    print(title)
    for row in rows:
        print(f"- rank={int(row['rank'])} image={row['image_path']}")
        print(f"  short: {row['acut_short_reason']}")
        print(f"  detailed: {row['acut_detailed_reason']}")
        if pd.notna(row.get("acut_comparison_reason")):
            print(f"  comparison: {row['acut_comparison_reason']}")
        if pd.notna(row.get("acut_rejection_reason")):
            print(f"  rejection: {row['acut_rejection_reason']}")


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    scores_csv, fallback_note = resolve_scores_csv(repo_root, args.scores_csv)
    vila_scores_csv = repo_root / args.vila_scores_csv
    if not vila_scores_csv.exists():
        raise FileNotFoundError(f"VILA scores CSV not found at {args.vila_scores_csv}")

    output_dir = repo_root / args.output_dir
    result = run_selector(
        repo_root=repo_root,
        scores_csv=scores_csv,
        vila_scores_csv=vila_scores_csv,
        output_dir=output_dir,
        top_k=args.top_k,
    )

    ranked_csv = output_dir / "ranked_results.csv"
    verification = verify_outputs(ranked_csv=ranked_csv, top_k=args.top_k)
    summary = {
        "requested_scores_csv": args.scores_csv,
        "resolved_scores_csv": str(scores_csv.relative_to(repo_root)),
        "vila_scores_csv": args.vila_scores_csv,
        "output_dir": str(output_dir.relative_to(repo_root)),
        "selector_stdout": result.stdout.strip(),
        "fallback_note": fallback_note,
        "verification": {
            key: value
            for key, value in verification.items()
            if key not in {"selected_examples", "rejected_examples"}
        },
    }
    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))
    print_examples("Selected examples", verification["selected_examples"])
    print_examples("Rejected examples", verification["rejected_examples"])


if __name__ == "__main__":
    main()
