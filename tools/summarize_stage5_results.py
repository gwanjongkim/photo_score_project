from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.infer.stage5_reference import ensure_project_path
from src.infer.stage5_review import write_stage5_review_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate lightweight review and product artifacts for a stage5 run.")
    parser.add_argument("run_dir", help="Existing stage5 output directory.")
    parser.add_argument("--top_k", type=int, help="Override the review top-k size.")
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_dir = ensure_project_path(args.run_dir)
    if run_dir is None or not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {args.run_dir}")

    artifacts = write_stage5_review_artifacts(run_dir, top_k=args.top_k, json_indent=args.json_indent)
    summary = {
        "run_dir": str(Path(run_dir)),
        **artifacts,
    }
    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
