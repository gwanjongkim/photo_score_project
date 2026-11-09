# TechIQA-Guard hard-FP 행을 학습 매니페스트에 추가하는 유틸리티.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TRAIN_CSV = "data/processed/techiqa_guard/train.csv"
DEFAULT_HARD_FP_CSV = "data/processed/techiqa_guard/hard_false_positive.csv"
DEFAULT_OUT_CSV = "data/processed/techiqa_guard/train_with_fp.csv"
FP_MARGIN_100 = 5.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append repeated TechIQA-Guard hard false-positive rows to train.csv."
    )
    parser.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--hard_fp_csv", default=DEFAULT_HARD_FP_CSV)
    parser.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--fp_repeat", type=int, default=8)
    parser.add_argument("--mos_weight_for_fp", type=float, default=0.0)
    parser.add_argument("--guard_weight_for_fp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df


def _first_available_guard_source(df: pd.DataFrame) -> pd.Series:
    guard = pd.Series(np.nan, index=df.index, dtype="float64")
    for column in ["existing_combined_score", "flive_teacher_score", "koniq_teacher_score"]:
        values = _numeric_column(df, column)
        guard = guard.where(guard.notna(), values)
    if "guard_source_score_100" in df.columns:
        existing_guard = pd.to_numeric(df["guard_source_score_100"], errors="coerce")
        guard = guard.where(guard.notna(), existing_guard)
    return guard


def _prepare_train_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = _ensure_columns(
        train_df,
        [
            "mos_loss_weight",
            "false_positive_loss_weight",
            "guard_source_score_100",
            "guard_cap_100",
            "false_positive_margin",
        ],
    )
    train_df["mos_loss_weight"] = _numeric_column(train_df, "mos_loss_weight").fillna(1.0)
    train_df["false_positive_loss_weight"] = _numeric_column(
        train_df, "false_positive_loss_weight"
    ).fillna(0.0)
    return train_df


def _prepare_hard_fp_rows(
    hard_fp_df: pd.DataFrame,
    fp_repeat: int,
    mos_weight_for_fp: float,
    guard_weight_for_fp: float,
    seed: int,
) -> pd.DataFrame:
    repeated = pd.concat([hard_fp_df.copy()] * fp_repeat, ignore_index=True)
    if len(repeated) > 0:
        repeated = repeated.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    repeated = _ensure_columns(
        repeated,
        [
            "hard_false_positive",
            "manual_false_positive",
            "mos_loss_weight",
            "false_positive_loss_weight",
            "guard_source_score_100",
            "guard_cap_100",
            "false_positive_margin",
        ],
    )

    repeated["hard_false_positive"] = True
    if "manual_false_positive" in hard_fp_df.columns:
        repeated["manual_false_positive"] = _bool_series(repeated["manual_false_positive"])
    elif "source_note" in repeated.columns:
        repeated["manual_false_positive"] = repeated["source_note"].astype(str).str.contains(
            "manual", case=False, na=False
        )
    else:
        repeated["manual_false_positive"] = False

    repeated["mos_loss_weight"] = float(mos_weight_for_fp)
    repeated["false_positive_loss_weight"] = float(guard_weight_for_fp)
    repeated["false_positive_margin"] = FP_MARGIN_100

    guard_source = _first_available_guard_source(repeated)
    repeated["guard_source_score_100"] = guard_source
    repeated["guard_cap_100"] = np.where(
        guard_source.notna(),
        guard_source + repeated["false_positive_margin"].astype("float64"),
        np.nan,
    )
    return repeated


def _hard_fp_count(df: pd.DataFrame) -> int:
    if "hard_false_positive" not in df.columns:
        return 0
    return int(_bool_series(df["hard_false_positive"]).sum())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.fp_repeat <= 0:
        raise ValueError("--fp_repeat must be positive.")
    if args.mos_weight_for_fp < 0.0:
        raise ValueError("--mos_weight_for_fp must be non-negative.")
    if args.guard_weight_for_fp < 0.0:
        raise ValueError("--guard_weight_for_fp must be non-negative.")

    train_path = Path(args.train_csv)
    hard_fp_path = Path(args.hard_fp_csv)
    out_path = Path(args.out_csv)
    summary_path = out_path.with_name("train_with_fp_summary.json")

    train_df = pd.read_csv(train_path)
    hard_fp_df = pd.read_csv(hard_fp_path)

    train_rows = _prepare_train_rows(train_df)
    hard_fp_rows = _prepare_hard_fp_rows(
        hard_fp_df=hard_fp_df,
        fp_repeat=args.fp_repeat,
        mos_weight_for_fp=args.mos_weight_for_fp,
        guard_weight_for_fp=args.guard_weight_for_fp,
        seed=args.seed,
    )

    output_df = pd.concat([train_rows, hard_fp_rows], ignore_index=True, sort=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)

    missing_guard_source = int(hard_fp_rows["guard_source_score_100"].isna().sum())
    summary = {
        "train_csv": str(train_path),
        "hard_fp_csv": str(hard_fp_path),
        "out_csv": str(out_path),
        "original_train_rows": int(len(train_df)),
        "hard_fp_rows": int(len(hard_fp_df)),
        "fp_repeat": int(args.fp_repeat),
        "appended_fp_rows": int(len(hard_fp_rows)),
        "final_rows": int(len(output_df)),
        "hard_fp_count_final": _hard_fp_count(output_df),
        "rows_missing_guard_source": missing_guard_source,
        "mos_weight_for_fp": float(args.mos_weight_for_fp),
        "guard_weight_for_fp": float(args.guard_weight_for_fp),
        "seed": int(args.seed),
    }
    _write_json(summary_path, summary)

    if missing_guard_source:
        print(
            f"WARNING: {missing_guard_source} appended hard-FP rows are missing guard-source scores.",
            file=sys.stderr,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
