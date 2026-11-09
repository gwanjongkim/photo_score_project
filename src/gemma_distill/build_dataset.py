from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.gemma_distill.common import (
    PROJECT_ROOT,
    append_jsonl,
    canonicalize_image_key,
    collapse_whitespace,
    ensure_mplconfigdir,
    read_json,
    repo_relative_path,
    resolve_repo_path,
    round_float,
    slugify,
    write_json,
    write_jsonl,
)
from src.gemma_distill.teacher import (
    build_rank_summary,
    build_training_messages,
    build_vila_teacher_prompt,
    compute_three_model_teacher_score,
    format_teacher_comment_3line,
    load_teacher_weight_config,
    validate_teacher_comment_3line,
)


DEFAULT_CONFIG = {
    "experiment_name": "gemma4_e4b_aesthetic3_smoke",
    "output_dir": "outputs/gemma4_e4b_aesthetic3_smoke",
    "teacher_weight_config": "configs/gemma4_e4b_aesthetic3_teacher.json",
    "teacher_manual_override_path": None,
    "teacher_runs": [],
    "teacher_generation": {
        "mode": "local_vila_worker",
        "vila_full_python": ".venv_vila/bin/python",
        "vila_full_model_path": "checkpoints/VILA1.5-3b",
        "vila_full_device": "cpu",
        "vila_full_max_new_tokens": 96,
        "fallback_to_existing_teacher_output": False,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Gemma 4 E4B distillation dataset with a strict 3-line VILA teacher.")
    parser.add_argument("--config", required=True)
    return parser


def log_event(log_path: Path, event: dict[str, Any]) -> None:
    append_jsonl(log_path, event)
    print(json.dumps(event, ensure_ascii=False))


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise ValueError("Could not resolve config path.")
    override = read_json(resolved)
    config = deep_merge(DEFAULT_CONFIG, override)
    return config, resolved


def score_lookup_from_frame(
    frame: pd.DataFrame,
    *,
    score_columns: dict[str, str],
    top_k: int,
    teacher_weight_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    working = frame.copy()
    working["__image_key"] = working["image_path"].map(canonicalize_image_key)
    working["nima_unit_score"] = None
    if score_columns.get("nima_unit_score"):
        working["nima_unit_score"] = pd.to_numeric(working[score_columns["nima_unit_score"]], errors="coerce")
    elif score_columns.get("nima_mean_score"):
        from src.gemma_distill.common import normalize_unit_score

        working["nima_unit_score"] = working[score_columns["nima_mean_score"]].map(normalize_unit_score)
    else:
        raise ValueError("Score columns must provide either nima_unit_score or nima_mean_score.")
    working["rgnet_score"] = pd.to_numeric(working[score_columns["rgnet_score"]], errors="coerce")
    working["alamp_score"] = pd.to_numeric(working[score_columns["alamp_score"]], errors="coerce")

    score_values: list[float | None] = []
    contribution_values: list[dict[str, Any]] = []
    for row in working.to_dict(orient="records"):
        weighted_score, contributions = compute_three_model_teacher_score(row, teacher_weight_config)
        score_values.append(weighted_score)
        contribution_values.append(contributions)
    working["weighted_aesthetic_score"] = score_values
    working["teacher_contributions_json"] = [json.dumps(payload, ensure_ascii=False, sort_keys=True) for payload in contribution_values]
    working = working.dropna(subset=["weighted_aesthetic_score", "__image_key"]).copy()
    working = working.sort_values(["weighted_aesthetic_score", "image_path"], ascending=[False, True]).reset_index(drop=True)
    working["rank"] = range(1, len(working) + 1)
    working["selected"] = working["rank"] <= int(top_k)
    working["rank_band"] = working["rank"].map(lambda rank: build_rank_summary(rank=rank, total_rows=len(working))["rank_band"])

    lookup: dict[str, dict[str, Any]] = {}
    for row in working.to_dict(orient="records"):
        lookup[str(row["__image_key"])] = {
            "nima_unit_score": round_float(row["nima_unit_score"]),
            "rgnet_score": round_float(row["rgnet_score"]),
            "alamp_score": round_float(row["alamp_score"]),
            "weighted_aesthetic_score": round_float(row["weighted_aesthetic_score"]),
            "teacher_contributions_json": row["teacher_contributions_json"],
            "rank": int(row["rank"]),
            "selected": bool(row["selected"]),
            "rank_band": row["rank_band"],
            "total_rows": int(len(working)),
        }
    return lookup


def stable_stratified_split(rows: list[dict[str, Any]], split_counts: dict[str, int]) -> list[dict[str, Any]]:
    selected_rows = [row for row in rows if bool(row.get("selected"))]
    rejected_rows = [row for row in rows if not bool(row.get("selected"))]
    selected_rows.sort(key=lambda row: (row.get("rank") is None, row.get("rank") or 10**9, str(row.get("image_path"))))
    rejected_rows.sort(key=lambda row: (row.get("rank") is None, row.get("rank") or 10**9, str(row.get("image_path"))))

    total_count = max(1, len(rows))
    target_counts = {str(key): int(value) for key, value in split_counts.items()}
    if sum(target_counts.values()) != len(rows):
        raise ValueError(
            "Split counts must sum to the number of rows. "
            f"Expected {len(rows)}, got {sum(target_counts.values())}."
        )

    group_targets = {
        True: {},
        False: {},
    }
    for split_name, split_count in target_counts.items():
        desired_selected = round((len(selected_rows) / total_count) * split_count)
        group_targets[True][split_name] = desired_selected
        group_targets[False][split_name] = split_count - desired_selected

    for is_selected, group_rows in ((True, selected_rows), (False, rejected_rows)):
        remaining = len(group_rows) - sum(group_targets[is_selected].values())
        split_names = list(target_counts)
        index = 0
        while remaining > 0:
            split_name = split_names[index % len(split_names)]
            group_targets[is_selected][split_name] += 1
            remaining -= 1
            index += 1
        while remaining < 0:
            split_name = split_names[index % len(split_names)]
            if group_targets[is_selected][split_name] > 0:
                group_targets[is_selected][split_name] -= 1
                remaining += 1
            index += 1

    split_rows = {split_name: [] for split_name in target_counts}
    for is_selected, group_rows in ((True, selected_rows), (False, rejected_rows)):
        cursor = 0
        for split_name in target_counts:
            count = group_targets[is_selected][split_name]
            taken = group_rows[cursor : cursor + count]
            for row in taken:
                split_rows[split_name].append(dict(row, split=split_name))
            cursor += count

    final_rows: list[dict[str, Any]] = []
    for split_name, split_group in split_rows.items():
        split_group.sort(key=lambda row: (row.get("rank") is None, row.get("rank") or 10**9, str(row.get("image_path"))))
        final_rows.extend(split_group)
    return final_rows


def extract_existing_teacher_output(row: dict[str, Any]) -> tuple[object, object, object]:
    return (
        row.get("multimodal_short_reason") or row.get("acut_short_reason"),
        row.get("multimodal_detailed_reason") or row.get("acut_detailed_reason"),
        row.get("multimodal_comparison_reason") or row.get("acut_comparison_reason"),
    )


def worker_result_is_usable(worker_row: dict[str, Any]) -> bool:
    raw_text = worker_row.get("raw_text") or worker_row.get("normalized_text")
    if validate_teacher_comment_3line(raw_text)["is_valid"]:
        return True

    parse_mode = str(worker_row.get("parse_mode") or "")
    short_reason = collapse_whitespace(worker_row.get("short_reason")) or ""
    detailed_reason = collapse_whitespace(worker_row.get("detailed_reason")) or ""
    if parse_mode not in {"strict_json", "embedded_json", "field_recovery"}:
        return False
    if not re.match(r"^(selected|rejected)\s*:", short_reason, flags=re.IGNORECASE):
        return False

    polluted_patterns = (
        r"\btechnical\b",
        r"\bfinal score\b",
        r"\branked\s*#",
        r"\bA-cut mainly because\b",
        r"\bimage is aesthetically pleasing\b",
        r"\bthis is not captioning\b",
    )
    combined = f"{short_reason} {detailed_reason}"
    return not any(re.search(pattern, combined, flags=re.IGNORECASE) for pattern in polluted_patterns)


def run_local_vila_worker(
    *,
    rows: list[dict[str, Any]],
    teacher_generation_cfg: dict[str, Any],
    output_dir: Path,
    log_path: Path,
) -> dict[str, dict[str, Any]]:
    input_payload = {
        "rows": [
            {
                "image_path": row["image_path"],
                "prompt": build_vila_teacher_prompt(row),
            }
            for row in rows
        ]
    }
    worker_input_path = write_json(output_dir / "vila_worker_input.json", input_payload)
    worker_stdout_path = output_dir / "vila_worker_stdout.txt"
    worker_json_path = output_dir / "vila_worker_output.json"
    vila_python = Path(str(teacher_generation_cfg["vila_full_python"]))
    if not vila_python.is_absolute():
        vila_python = PROJECT_ROOT / vila_python
    vila_model_path = Path(str(teacher_generation_cfg["vila_full_model_path"]))
    if not vila_model_path.is_absolute():
        vila_model_path = PROJECT_ROOT / vila_model_path

    command = [
        str(vila_python),
        "-m",
        "src.infer.vila_full_local_worker",
        "--model_path",
        str(vila_model_path),
        "--input_json",
        str(worker_input_path),
        "--device",
        str(teacher_generation_cfg.get("vila_full_device", "cpu")),
        "--max_new_tokens",
        str(int(teacher_generation_cfg.get("vila_full_max_new_tokens", 96))),
    ]
    log_event(
        log_path,
        {
            "phase": "local_vila_worker_start",
            "command": command,
            "row_count": len(rows),
        },
    )
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    worker_stdout_path.write_text((completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Local VILA worker failed with exit code {completed.returncode}. "
            f"See {worker_stdout_path}."
        )

    stdout_text = completed.stdout.strip()
    start_index = stdout_text.find("{")
    if start_index < 0:
        raise RuntimeError(f"Local VILA worker output did not contain JSON. See {worker_stdout_path}.")
    payload = json.loads(stdout_text[start_index:])
    write_json(worker_json_path, payload)

    results_by_key: dict[str, dict[str, Any]] = {}
    for result in payload.get("results") or []:
        image_key = canonicalize_image_key(result.get("image_path"))
        if image_key is None:
            continue
        results_by_key[image_key] = result
    log_event(
        log_path,
        {
            "phase": "local_vila_worker_complete",
            "row_count": len(rows),
            "parsed_results": len(results_by_key),
            "worker_stdout_path": repo_relative_path(worker_stdout_path),
        },
    )
    return results_by_key


def build_rows_from_teacher_run(
    run_cfg: dict[str, Any],
    *,
    teacher_weight_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    teacher_rows = read_json(run_cfg["path"])
    if not isinstance(teacher_rows, list):
        raise ValueError(f"Teacher run must be a JSON list: {run_cfg['path']}")
    score_frame = pd.read_csv(resolve_repo_path(run_cfg["score_csv"]))
    score_lookup = score_lookup_from_frame(
        score_frame,
        score_columns=dict(run_cfg["score_columns"]),
        top_k=int(run_cfg["top_k"]),
        teacher_weight_config=teacher_weight_config,
    )

    rows: list[dict[str, Any]] = []
    for teacher_row in teacher_rows:
        image_key = canonicalize_image_key(teacher_row.get("image_path"))
        if image_key is None or image_key not in score_lookup:
            continue
        score_info = dict(score_lookup[image_key])
        merged = {
            "dataset": str(run_cfg["dataset"]),
            "image_path": str(resolve_repo_path(teacher_row["image_path"])),
            "image_file_name": Path(str(teacher_row["image_path"])).name,
            "source_teacher_run": repo_relative_path(run_cfg["path"]),
            "source_score_csv": repo_relative_path(run_cfg["score_csv"]),
            "source_teacher_selected": bool(teacher_row.get("selected")),
            "selected": bool(score_info["selected"]),
            "rank": int(score_info["rank"]),
            "rank_band": score_info["rank_band"],
            "nima_unit_score": score_info["nima_unit_score"],
            "rgnet_score": score_info["rgnet_score"],
            "alamp_score": score_info["alamp_score"],
            "weighted_aesthetic_score": score_info["weighted_aesthetic_score"],
            "teacher_contributions_json": score_info["teacher_contributions_json"],
            "source_existing_short_reason": extract_existing_teacher_output(teacher_row)[0],
            "source_existing_detailed_reason": extract_existing_teacher_output(teacher_row)[1],
            "source_existing_comparison_reason": extract_existing_teacher_output(teacher_row)[2],
        }
        rows.append(merged)

    rows.sort(key=lambda row: (row["rank"], row["image_path"]))
    if run_cfg.get("split"):
        rows = [dict(row, split=str(run_cfg["split"])) for row in rows]
    elif run_cfg.get("split_counts"):
        rows = stable_stratified_split(rows, split_counts=dict(run_cfg["split_counts"]))
    else:
        raise ValueError(f"Teacher run must define either split or split_counts: {run_cfg['path']}")
    return rows, score_frame


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1
        selected_key = "selected" if row["selected"] else "rejected"
        selected_counts[selected_key] = selected_counts.get(selected_key, 0) + 1
    return {
        "rows": len(rows),
        "split_counts": split_counts,
        "dataset_counts": dataset_counts,
        "selected_counts": selected_counts,
    }


def main() -> None:
    args = build_parser().parse_args()
    ensure_mplconfigdir()

    config, config_path = load_config(args.config)
    output_dir = resolve_repo_path(config["output_dir"])
    if output_dir is None:
        raise ValueError("Could not resolve output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "build_log.jsonl"

    teacher_weight_config, teacher_weight_config_path = load_teacher_weight_config(config["teacher_weight_config"])
    teacher_manual_overrides: dict[str, str] = {}
    if config.get("teacher_manual_override_path"):
        override_path = resolve_repo_path(config["teacher_manual_override_path"])
        if override_path is None or not override_path.exists():
            raise FileNotFoundError(f"Teacher override file not found: {config['teacher_manual_override_path']}")
        teacher_manual_overrides = {
            str(key): str(value)
            for key, value in dict(read_json(override_path)).items()
            if str(value).strip()
        }
    write_json(output_dir / "resolved_dataset_config.json", config)
    write_json(output_dir / "resolved_teacher_weight_config.json", teacher_weight_config)

    log_event(
        log_path,
        {
            "phase": "dataset_build_start",
            "config_path": repo_relative_path(config_path),
            "teacher_weight_config_path": repo_relative_path(teacher_weight_config_path),
            "output_dir": repo_relative_path(output_dir),
        },
    )

    all_rows: list[dict[str, Any]] = []
    score_sources_summary: list[dict[str, Any]] = []
    for teacher_run_cfg in config["teacher_runs"]:
        rows, score_frame = build_rows_from_teacher_run(
            teacher_run_cfg,
            teacher_weight_config=teacher_weight_config,
        )
        all_rows.extend(rows)
        score_sources_summary.append(
            {
                "dataset": teacher_run_cfg["dataset"],
                "teacher_run_path": repo_relative_path(teacher_run_cfg["path"]),
                "score_csv": repo_relative_path(teacher_run_cfg["score_csv"]),
                "rows_used": len(rows),
                "source_score_rows": int(len(score_frame)),
                "top_k": int(teacher_run_cfg["top_k"]),
            }
        )

    all_rows.sort(key=lambda row: (row["split"], row["dataset"], row["rank"], row["image_path"]))
    pre_generation_summary = summarize_rows(all_rows)
    write_json(output_dir / "score_sources_summary.json", score_sources_summary)
    write_json(output_dir / "pre_generation_summary.json", pre_generation_summary)
    log_event(log_path, {"phase": "teacher_rows_loaded", **pre_generation_summary})

    teacher_generation_cfg = dict(config["teacher_generation"])
    teacher_generation_mode = str(teacher_generation_cfg.get("mode") or "local_vila_worker")
    worker_results: dict[str, dict[str, Any]] = {}
    only_splits = {str(value) for value in teacher_generation_cfg.get("only_splits") or []}
    if teacher_generation_mode == "local_vila_worker":
        generation_rows = [row for row in all_rows if not only_splits or str(row["split"]) in only_splits]
        worker_results = run_local_vila_worker(
            rows=generation_rows,
            teacher_generation_cfg=teacher_generation_cfg,
            output_dir=output_dir,
            log_path=log_path,
        )
    elif teacher_generation_mode != "reuse_existing_teacher_output":
        raise ValueError(f"Unknown teacher_generation.mode: {teacher_generation_mode}")

    dataset_rows: list[dict[str, Any]] = []
    worker_attempted_row_count = len(worker_results)
    worker_accepted_row_count = 0
    for row in all_rows:
        image_key = canonicalize_image_key(row["image_path"])
        worker_row = worker_results.get(image_key) if image_key is not None else None
        if worker_row is not None and not worker_row.get("error") and worker_result_is_usable(worker_row):
            worker_accepted_row_count += 1
            raw_teacher_comment = worker_row.get("raw_text") or worker_row.get("normalized_text")
            raw_validation = validate_teacher_comment_3line(raw_teacher_comment)
            if raw_validation["is_valid"]:
                teacher_comment = "\n".join(raw_validation["lines"])
            else:
                teacher_comment = format_teacher_comment_3line(
                    short_reason=worker_row.get("short_reason"),
                    detailed_reason=worker_row.get("detailed_reason"),
                    comparison_reason=worker_row.get("comparison_reason"),
                    selected=bool(row["selected"]),
                )
            teacher_model_name = teacher_generation_cfg.get("vila_full_model_path")
            teacher_source_mode = "local_vila_worker"
            raw_teacher_fields = worker_row
        else:
            if not bool(teacher_generation_cfg.get("fallback_to_existing_teacher_output", False)):
                continue
            if bool(row.get("source_teacher_selected")) != bool(row.get("selected")):
                continue
            teacher_comment = format_teacher_comment_3line(
                short_reason=row.get("source_existing_short_reason"),
                detailed_reason=row.get("source_existing_detailed_reason"),
                comparison_reason=row.get("source_existing_comparison_reason"),
                selected=bool(row["selected"]),
            )
            teacher_model_name = "existing_vila_full_local_output"
            teacher_source_mode = "existing_teacher_output_fallback"
            raw_teacher_fields = {
                "short_reason": row.get("source_existing_short_reason"),
                "detailed_reason": row.get("source_existing_detailed_reason"),
                "comparison_reason": row.get("source_existing_comparison_reason"),
            }

        validation = validate_teacher_comment_3line(teacher_comment)
        if not validation["is_valid"]:
            continue

        row_id = f"{slugify(row['dataset'])}__{slugify(row['split'])}__{slugify(Path(row['image_path']).name)}"
        override_comment = teacher_manual_overrides.get(row_id)
        if override_comment:
            override_validation = validate_teacher_comment_3line(override_comment)
            if not override_validation["is_valid"]:
                raise ValueError(f"Teacher override for {row_id} is not a valid 3-line comment.")
            teacher_comment = "\n".join(override_validation["lines"])
            validation = override_validation
        enriched = dict(row)
        enriched.update(
            {
                "id": row_id,
                "image": row["image_path"],
                "teacher_model_name": str(teacher_model_name),
                "teacher_source_mode": teacher_source_mode,
                "teacher_comment_3line": teacher_comment,
                "teacher_comment_strength_line": validation["lines"][0],
                "teacher_comment_weakness_line": validation["lines"][1],
                "teacher_comment_verdict_line": validation["lines"][2],
                "teacher_format_valid": validation["is_valid"],
                "teacher_raw_fields_json": json.dumps(raw_teacher_fields, ensure_ascii=False),
            }
        )
        enriched["messages"] = build_training_messages(enriched, teacher_comment_3line=teacher_comment)
        dataset_rows.append(enriched)

    dataset_rows.sort(key=lambda row: (row["split"], row["dataset"], row["rank"], row["image_path"]))
    teacher_source_mode_counts: dict[str, int] = {}
    for row in dataset_rows:
        teacher_source_mode_counts[row["teacher_source_mode"]] = teacher_source_mode_counts.get(row["teacher_source_mode"], 0) + 1

    manifest_csv_rows = []
    jsonl_rows = []
    for row in dataset_rows:
        manifest_csv_rows.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "messages",
                }
            }
        )
        jsonl_rows.append(
            {
                "id": row["id"],
                "image": row["image"],
                "image_path": row["image_path"],
                "split": row["split"],
                "dataset": row["dataset"],
                "selected": row["selected"],
                "rank": row["rank"],
                "rank_band": row["rank_band"],
                "nima_unit_score": row["nima_unit_score"],
                "rgnet_score": row["rgnet_score"],
                "alamp_score": row["alamp_score"],
                "weighted_aesthetic_score": row["weighted_aesthetic_score"],
                "teacher_comment_3line": row["teacher_comment_3line"],
                "teacher_comment_strength_line": row["teacher_comment_strength_line"],
                "teacher_comment_weakness_line": row["teacher_comment_weakness_line"],
                "teacher_comment_verdict_line": row["teacher_comment_verdict_line"],
                "teacher_model_name": row["teacher_model_name"],
                "teacher_source_mode": row["teacher_source_mode"],
                "teacher_contributions_json": row["teacher_contributions_json"],
                "messages": row["messages"],
            }
        )

    manifest_frame = pd.DataFrame(manifest_csv_rows)
    manifest_csv_path = output_dir / "dataset_manifest.csv"
    manifest_frame.to_csv(manifest_csv_path, index=False)
    write_jsonl(output_dir / "dataset_all.jsonl", jsonl_rows)

    split_output_paths: dict[str, str] = {}
    for split_name, split_rows in manifest_frame.groupby("split", sort=True):
        split_name = str(split_name)
        split_csv_path = output_dir / f"{split_name}.csv"
        split_jsonl_path = output_dir / f"{split_name}.jsonl"
        split_rows.to_csv(split_csv_path, index=False)
        split_jsonl_rows = [row for row in jsonl_rows if row["split"] == split_name]
        write_jsonl(split_jsonl_path, split_jsonl_rows)
        split_output_paths[split_name] = repo_relative_path(split_jsonl_path)

    format_checks = [validate_teacher_comment_3line(row["teacher_comment_3line"]) for row in dataset_rows]
    valid_count = sum(1 for item in format_checks if item["is_valid"])
    format_summary = {
        "rows": len(dataset_rows),
        "valid_three_line_count": valid_count,
        "valid_three_line_rate": round(valid_count / len(dataset_rows), 6) if dataset_rows else 0.0,
        "prefix_ok_rate": round(sum(1 for item in format_checks if item["prefix_ok"]) / len(dataset_rows), 6)
        if dataset_rows
        else 0.0,
        "role_separation_ok_rate": round(sum(1 for item in format_checks if item["role_separation_ok"]) / len(dataset_rows), 6)
        if dataset_rows
        else 0.0,
    }
    write_json(output_dir / "teacher_format_summary.json", format_summary)

    dataset_summary = {
        "experiment_name": config["experiment_name"],
        "output_dir": repo_relative_path(output_dir),
        "config_path": repo_relative_path(config_path),
        "teacher_weight_config_path": repo_relative_path(teacher_weight_config_path),
        "teacher_generation_mode": teacher_generation_mode,
        "teacher_generation_model": teacher_generation_cfg.get("vila_full_model_path"),
        "teacher_generation_only_splits": sorted(only_splits),
        "teacher_worker_attempted_row_count": worker_attempted_row_count,
        "teacher_worker_accepted_row_count": worker_accepted_row_count,
        "source_runs": score_sources_summary,
        "rows_before_teacher_generation": pre_generation_summary["rows"],
        "rows_after_teacher_generation": len(dataset_rows),
        "split_counts": {str(key): int(value) for key, value in manifest_frame["split"].value_counts().to_dict().items()},
        "dataset_counts": {str(key): int(value) for key, value in manifest_frame["dataset"].value_counts().to_dict().items()},
        "selected_counts": {
            "selected": int(manifest_frame["selected"].astype(bool).sum()),
            "rejected": int((~manifest_frame["selected"].astype(bool)).sum()),
        }
        if not manifest_frame.empty
        else {"selected": 0, "rejected": 0},
        "teacher_source_mode_counts": teacher_source_mode_counts,
        "split_output_paths": split_output_paths,
        "teacher_format_summary": format_summary,
    }
    write_json(output_dir / "dataset_summary.json", dataset_summary)
    log_event(log_path, {"phase": "dataset_build_complete", **dataset_summary})


if __name__ == "__main__":
    main()
