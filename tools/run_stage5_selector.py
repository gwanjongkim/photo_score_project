from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.infer.stage5_reference import (
    default_stage5_output_dir,
    ensure_project_path,
    load_stage5_reference_config,
    repo_relative_path,
)
from src.infer.stage5_review import write_stage5_review_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen stage5 reference best-shot selector.")
    parser.add_argument("--input_dir", required=True, help="Folder of candidate images to rank.")
    parser.add_argument(
        "--output_dir",
        help="Destination directory for this run. Defaults to outputs/stage5_runs/<label>_<timestamp>.",
    )
    parser.add_argument("--run_label", help="Human-readable run label used when auto-generating the output directory.")
    parser.add_argument(
        "--config",
        default="configs/stage5_reference.json",
        help="Stage5 preset config to load.",
    )
    parser.add_argument("--top_k", type=int, help="Override the preset top-k size.")
    parser.add_argument("--score_recursive", action="store_true", help="Score images recursively inside the input folder.")
    parser.add_argument("--score_extensions", help="Comma-separated extensions to score.")
    parser.add_argument("--copy_top_k", action="store_true", help="Copy the top-k images into output_dir/top_k.")
    parser.add_argument("--symlink_top_k", action="store_true", help="Symlink the top-k images into output_dir/top_k.")
    parser.add_argument("--weights_config", help="Optional selector weight override JSON.")
    parser.add_argument("--skip_review", action="store_true", help="Skip the extra review/product artifact export step.")
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def normalize_output_dir(path: str | Path | None, input_dir: Path, run_label: str | None) -> Path:
    if path:
        resolved = ensure_project_path(path)
        if resolved is None:
            raise ValueError("Could not resolve output_dir.")
        return resolved
    return default_stage5_output_dir(input_dir=input_dir, run_label=run_label)


def ensure_clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_runtime_environment() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/photo_score_project_mplconfig")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def serialize_mapping(mapping: dict[str, object]) -> dict[str, object]:
    serialized = {}
    for key, value in mapping.items():
        if isinstance(value, str) and (
            key.endswith("_model") or key.endswith("_csv") or key.endswith("_dir") or key.endswith("_config")
        ):
            serialized[key] = repo_relative_path(value)
        else:
            serialized[key] = value
    return serialized


def build_runtime_namespace(
    cli_args: argparse.Namespace,
    config: dict[str, object],
    input_dir: Path,
    output_dir: Path,
) -> argparse.Namespace:
    bundle_args = dict(config.get("bundle_args", {}))
    selector_args = dict(config.get("selector_args", {}))
    weights_config = ensure_project_path(cli_args.weights_config) if cli_args.weights_config else None
    return argparse.Namespace(
        input_dir=str(input_dir),
        scores_jsonl=None,
        scores_csv=None,
        output_dir=str(output_dir),
        top_k=int(cli_args.top_k or selector_args.get("top_k", 5)),
        copy_top_k=bool(cli_args.copy_top_k),
        symlink_top_k=bool(cli_args.symlink_top_k),
        weights_config=str(weights_config) if weights_config else None,
        enable_pairwise_rerank=bool(selector_args.get("enable_pairwise_rerank", True)),
        rerank_pool_size=int(selector_args.get("rerank_pool_size", 10)),
        enable_diversity=bool(selector_args.get("enable_diversity", True)),
        diversity_threshold=float(selector_args.get("diversity_threshold", 0.82)),
        hard_duplicate_threshold=float(selector_args.get("hard_duplicate_threshold", 0.88)),
        diversity_penalty_strength=float(selector_args.get("diversity_penalty_strength", 0.12)),
        score_recursive=bool(cli_args.score_recursive or selector_args.get("score_recursive", False)),
        score_extensions=str(cli_args.score_extensions or selector_args.get("score_extensions", ".jpg,.jpeg,.png,.webp,.bmp")),
        include_debug=False,
        alamp_include_debug=False,
        json_indent=int(cli_args.json_indent),
        **bundle_args,
    )


def main() -> None:
    parser = build_parser()
    cli_args = parser.parse_args()
    if cli_args.copy_top_k and cli_args.symlink_top_k:
        raise ValueError("Choose only one of --copy_top_k or --symlink_top_k.")

    input_dir = ensure_project_path(cli_args.input_dir)
    if input_dir is None or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {cli_args.input_dir}")

    config, config_path = load_stage5_reference_config(cli_args.config)
    output_dir = normalize_output_dir(cli_args.output_dir, input_dir=input_dir, run_label=cli_args.run_label)
    ensure_clean_output_dir(output_dir)
    prepare_runtime_environment()

    from src.infer.select_best_shots import (
        build_diversity_config,
        load_or_compute_scores,
        load_weights,
        materialize_top_k,
        rank_records,
        write_duplicate_groups_report,
        write_ranked_outputs,
    )

    runtime_args = build_runtime_namespace(cli_args, config=config, input_dir=input_dir, output_dir=output_dir)
    weights = load_weights(runtime_args.weights_config)
    records, source_jsonl, source_csv = load_or_compute_scores(runtime_args, output_dir)
    diversity_config = build_diversity_config(runtime_args) if runtime_args.enable_diversity else None
    ranked_rows, duplicate_groups_report = rank_records(
        records,
        weights=weights,
        enable_pairwise_rerank=runtime_args.enable_pairwise_rerank,
        rerank_pool_size=runtime_args.rerank_pool_size,
        diversity_config=diversity_config,
    )

    ranked_jsonl, ranked_csv, top_k_list = write_ranked_outputs(ranked_rows, output_dir=output_dir, top_k=runtime_args.top_k)
    duplicate_groups_json = write_duplicate_groups_report(duplicate_groups_report, output_dir=output_dir)
    top_k_dir = materialize_top_k(
        ranked_rows,
        output_dir=output_dir,
        top_k=runtime_args.top_k,
        copy_top_k=runtime_args.copy_top_k,
        symlink_top_k=runtime_args.symlink_top_k,
    )

    manifest = {
        "pipeline_name": config.get("pipeline_name"),
        "pipeline_version": config.get("pipeline_version"),
        "description": config.get("description"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": cli_args.run_label or output_dir.name,
        "config_path": repo_relative_path(config_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_images": len(ranked_rows),
        "top_k": runtime_args.top_k,
        "selector_args": serialize_mapping(
            {
                "top_k": runtime_args.top_k,
                "enable_pairwise_rerank": runtime_args.enable_pairwise_rerank,
                "rerank_pool_size": runtime_args.rerank_pool_size,
                "enable_diversity": runtime_args.enable_diversity,
                "diversity_threshold": runtime_args.diversity_threshold,
                "hard_duplicate_threshold": runtime_args.hard_duplicate_threshold,
                "diversity_penalty_strength": runtime_args.diversity_penalty_strength,
                "score_recursive": runtime_args.score_recursive,
                "score_extensions": runtime_args.score_extensions,
                "weights_config": runtime_args.weights_config,
            }
        ),
        "bundle_args": serialize_mapping(dict(config.get("bundle_args", {}))),
        "artifacts": {
            "scores_jsonl": repo_relative_path(source_jsonl),
            "scores_csv": repo_relative_path(source_csv),
            "ranked_jsonl": repo_relative_path(ranked_jsonl),
            "ranked_csv": repo_relative_path(ranked_csv),
            "duplicate_groups_json": repo_relative_path(duplicate_groups_json),
            "top_k_list": repo_relative_path(top_k_list),
            "top_k_dir": repo_relative_path(top_k_dir),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=cli_args.json_indent, ensure_ascii=False) + "\n", encoding="utf-8")

    review_artifacts = {}
    if not cli_args.skip_review:
        review_artifacts = write_stage5_review_artifacts(
            output_dir=output_dir,
            top_k=runtime_args.top_k,
            json_indent=cli_args.json_indent,
        )

    summary = {
        "pipeline_name": manifest["pipeline_name"],
        "pipeline_version": manifest["pipeline_version"],
        "run_label": manifest["run_label"],
        "input_dir": manifest["input_dir"],
        "output_dir": manifest["output_dir"],
        "num_images": len(ranked_rows),
        "top_k": runtime_args.top_k,
        "run_manifest": str(manifest_path),
        "artifacts": manifest["artifacts"],
        "review_artifacts": review_artifacts,
    }
    print(json.dumps(summary, indent=cli_args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
