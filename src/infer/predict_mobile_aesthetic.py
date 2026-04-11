from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.infer.predict_quality_bundle import add_bundle_arguments, load_bundle_models, predict_bundle_for_image
from src.infer.score_image_folder import iter_image_paths
from src.infer.select_best_shots import merge_nested_dict, score_component


DEFAULT_MOBILE_POLICY = {
    "default_mobile": {
        "component_mix": {
            "aesthetic": 0.55,
            "technical": 0.45,
        },
        "aesthetic": {
            "aadb_score": 1.0,
        },
        "technical": {
            "koniq_score": 0.60,
            "flive_image_score": 0.40,
        },
    },
    "extended": {
        "component_mix": {
            "aesthetic": 0.55,
            "technical": 0.45,
        },
        "aesthetic": {
            "aadb_score": 0.55,
            "nima_mean_score": 0.30,
            "alamp_score": 0.10,
            "rgnet_score": 0.05,
        },
        "technical": {
            "koniq_score": 0.45,
            "flive_image_score": 0.30,
            "flive_patch_mean": 0.15,
            "musiq_score": 0.10,
        },
    },
}


DEFAULT_CHECKPOINT_CANDIDATES = {
    "aadb_model": [
        "checkpoints/composition_aadb_gpu/final_model.keras",
    ],
    "koniq_model": [
        "checkpoints/technical_koniq_gpu/final_model.keras",
    ],
    "flive_image_model": [
        "checkpoints/technical_flive_image_gpu/final_model.keras",
    ],
    "flive_patch_model": [
        "checkpoints/technical_flive_patch_gpu/final_model.keras",
    ],
    "nima_model": [
        "checkpoints/nima_ava_gpu/final_model.keras",
        "outputs/nima_ava_restart/final_model.keras",
    ],
    "alamp_model": [
        "checkpoints/alamp_aadb_gpu/final_model.keras",
        "outputs/alamp_restart/final_model.keras",
    ],
    "musiq_model": [
        "checkpoints/musiq_aadb_gpu/final_model.keras",
        "outputs/musiq_restart/final_model.keras",
    ],
    "rgnet_model": [
        "checkpoints/rgnet_aadb_gpu/final_model.keras",
        "outputs/rgnet_restart/final_model.keras",
    ],
}


ON_DEVICE_MODEL_FIELDS = {
    "aadb_model": "aadb",
    "koniq_model": "koniq",
    "flive_image_model": "flive_image",
}


EXTENDED_MODEL_FIELDS = {
    "nima_model": "nima",
    "flive_patch_model": "flive_patch",
    "alamp_model": "alamp",
    "musiq_model": "musiq",
    "rgnet_model": "rgnet",
}


MODEL_SCORE_KEYS = {
    "aadb": "aadb_score",
    "koniq": "koniq_score",
    "flive_image": "flive_image_score",
    "flive_patch": "flive_patch_mean",
    "nima": "nima_mean_score",
    "alamp": "alamp_score",
    "musiq": "musiq_score",
    "rgnet": "rgnet_score",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a mobile-oriented photo aesthetic evaluation pipeline using the repo's trained models."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image_path")
    source_group.add_argument("--input_dir")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.webp,.bmp")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Add the optional richer local/offline models when checkpoints are available.",
    )
    parser.add_argument(
        "--no_repo_defaults",
        action="store_true",
        help="Require explicit model paths instead of auto-resolving repo checkpoints.",
    )
    parser.add_argument("--policy_config")
    parser.add_argument("--output_json")
    parser.add_argument("--top_k", type=int, default=5)
    return add_bundle_arguments(parser)


def load_mobile_policy(policy_config: str | None) -> dict[str, object]:
    policy = json.loads(json.dumps(DEFAULT_MOBILE_POLICY))
    if not policy_config:
        return policy
    with Path(policy_config).open("r", encoding="utf-8") as f:
        override = json.load(f)
    merge_nested_dict(policy, override)
    return policy


def resolve_repo_checkpoint(field_name: str) -> str | None:
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES.get(field_name, []):
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def resolve_bundle_args(args: argparse.Namespace) -> tuple[argparse.Namespace, dict[str, object]]:
    bundle_args = argparse.Namespace(**vars(args))
    pipeline_notes = []
    warnings = []
    resolved_paths = {}

    if args.pairwise_model or args.pairwise_reference_csv:
        pipeline_notes.append("Pairwise inputs were ignored in the mobile scorer; keep pairwise for post-hoc reranking only.")
    bundle_args.pairwise_model = None
    bundle_args.pairwise_reference_csv = None

    for field_name in ON_DEVICE_MODEL_FIELDS:
        value = getattr(args, field_name)
        if value is None and not args.no_repo_defaults:
            value = resolve_repo_checkpoint(field_name)
            if value is not None:
                pipeline_notes.append(f"Resolved repo default for {field_name}: {value}")
        setattr(bundle_args, field_name, value)
        if value is not None:
            resolved_paths[field_name] = value

    for field_name in EXTENDED_MODEL_FIELDS:
        value = getattr(args, field_name)
        if args.extended:
            if value is None and not args.no_repo_defaults:
                value = resolve_repo_checkpoint(field_name)
                if value is not None:
                    pipeline_notes.append(f"Resolved repo default for {field_name}: {value}")
            if value is None:
                warnings.append(f"Optional extended model unavailable: {field_name}")
        else:
            if value is not None:
                pipeline_notes.append(f"Ignored {field_name} because --extended was not enabled.")
            value = None
        setattr(bundle_args, field_name, value)
        if value is not None:
            resolved_paths[field_name] = value

    if not any(getattr(bundle_args, field_name) for field_name in ON_DEVICE_MODEL_FIELDS):
        raise ValueError("No on-device model checkpoints were available. Provide explicit model paths or enable repo defaults.")

    return bundle_args, {
        "pipeline_notes": pipeline_notes,
        "warnings": warnings,
        "resolved_paths": resolved_paths,
    }


def combine_component_scores(component_mix: dict[str, float], aesthetic_score: float | None, technical_score: float | None) -> float | None:
    if aesthetic_score is not None and technical_score is not None:
        return float(component_mix["aesthetic"] * aesthetic_score + component_mix["technical"] * technical_score)
    if aesthetic_score is not None:
        return float(aesthetic_score)
    if technical_score is not None:
        return float(technical_score)
    return None


def gather_used_models(record: dict[str, object], model_fields: dict[str, str]) -> list[str]:
    used = []
    for family in model_fields.values():
        score_key = MODEL_SCORE_KEYS[family]
        if record.get(score_key) is not None:
            used.append(family)
    return used


def score_mobile_record(
    record: dict[str, object],
    policy: dict[str, object],
    extended: bool,
    pipeline_meta: dict[str, object],
) -> dict[str, object]:
    policy_name = "extended" if extended else "default_mobile"
    policy_block = policy[policy_name]

    aesthetic_score, aesthetic_contrib, aesthetic_missing = score_component(
        record,
        weights=policy_block["aesthetic"],
        component_name="aesthetic",
    )
    technical_score, technical_contrib, technical_missing = score_component(
        record,
        weights=policy_block["technical"],
        component_name="technical",
    )
    final_score = combine_component_scores(
        component_mix=policy_block["component_mix"],
        aesthetic_score=aesthetic_score,
        technical_score=technical_score,
    )

    notes = []
    warnings = []
    if extended:
        if gather_used_models(record, EXTENDED_MODEL_FIELDS):
            notes.append("Extended mode added higher-cost models to refine the score.")
        else:
            notes.append("Extended mode requested, but no optional extended models were available. Mobile-only scoring was used.")
    else:
        notes.append("Default mobile mode used only the lightweight on-device scorer subset.")

    if aesthetic_score is None and technical_score is None:
        warnings.append("No usable model scores were available for mobile fusion.")
    elif aesthetic_score is None:
        warnings.append("Aesthetic models were unavailable; final score falls back to technical-only scoring.")
    elif technical_score is None:
        warnings.append("Technical models were unavailable; final score falls back to aesthetic-only scoring.")

    if pipeline_meta["warnings"]:
        warnings.extend(str(item) for item in pipeline_meta["warnings"])

    return {
        "image_path": record.get("image_path"),
        "mode": policy_name,
        "final_score": final_score,
        "aesthetic_score": aesthetic_score,
        "technical_score": technical_score,
        "on_device_models_used": gather_used_models(record, ON_DEVICE_MODEL_FIELDS),
        "optional_extended_models_used": gather_used_models(record, EXTENDED_MODEL_FIELDS),
        "per_model_scores": {
            key: record.get(key)
            for key in [
                "aadb_score",
                "koniq_score",
                "flive_image_score",
                "flive_patch_mean",
                "nima_mean_score",
                "alamp_score",
                "musiq_score",
                "rgnet_score",
                "baseline_final_score",
            ]
            if record.get(key) is not None
        },
        "component_details": {
            "aesthetic": aesthetic_contrib,
            "technical": technical_contrib,
        },
        "missing_mobile_aesthetic_models": sorted(set(aesthetic_missing)),
        "missing_mobile_technical_models": sorted(set(technical_missing)),
        "notes": notes,
        "warnings": warnings,
    }


def score_paths_for_mobile(
    image_paths: list[Path],
    bundle_args: argparse.Namespace,
    policy: dict[str, object],
    extended: bool,
    pipeline_meta: dict[str, object],
) -> list[dict[str, object]]:
    models = load_bundle_models(bundle_args)
    results = []
    for image_path in image_paths:
        bundle_result = predict_bundle_for_image(image_path, args=bundle_args, models=models)
        results.append(
            score_mobile_record(
                bundle_result,
                policy=policy,
                extended=extended,
                pipeline_meta=pipeline_meta,
            )
        )
    results.sort(
        key=lambda row: (
            row["final_score"] is not None,
            float("-inf") if row["final_score"] is None else float(row["final_score"]),
        ),
        reverse=True,
    )
    for index, row in enumerate(results, start=1):
        row["rank"] = index
    return results


def build_output_payload(
    args: argparse.Namespace,
    results: list[dict[str, object]],
    pipeline_meta: dict[str, object],
) -> dict[str, object]:
    top_k = results[: args.top_k]
    common = {
        "mode": "extended" if args.extended else "default_mobile",
        "pipeline_notes": pipeline_meta["pipeline_notes"],
        "resolved_model_paths": pipeline_meta["resolved_paths"],
    }

    if args.image_path:
        return {
            **common,
            **results[0],
        }

    return {
        **common,
        "input_dir": str(args.input_dir),
        "num_images": len(results),
        "top_k": top_k,
        "results": results,
    }


def write_output_json(payload: dict[str, object], output_json: str | None) -> None:
    if not output_json:
        return
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    policy = load_mobile_policy(args.policy_config)
    bundle_args, pipeline_meta = resolve_bundle_args(args)

    if args.image_path:
        image_paths = [Path(args.image_path)]
    else:
        extensions = {part.strip().lower() for part in args.extensions.split(",") if part.strip()}
        image_paths = iter_image_paths(Path(args.input_dir), recursive=args.recursive, extensions=extensions)
        if not image_paths:
            raise ValueError("No images matched the provided folder and extension filters.")

    results = score_paths_for_mobile(
        image_paths=image_paths,
        bundle_args=bundle_args,
        policy=policy,
        extended=args.extended,
        pipeline_meta=pipeline_meta,
    )
    payload = build_output_payload(args, results=results, pipeline_meta=pipeline_meta)
    write_output_json(payload, args.output_json)
    print(json.dumps(payload, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
