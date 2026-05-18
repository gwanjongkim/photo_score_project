# TOPIQ-lite 순위 학습용 FLIVE 쌍 데이터셋을 구성한다.
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from src.datasets.arp_dataset import make_arp_iqa_dataset


AUTOTUNE = tf.data.AUTOTUNE


def _resolve_csv_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_image_path(value: Any, csv_dir: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = csv_dir / path
    return str(path.resolve())


def load_iqa_csv(
    csv_path: str | Path,
    target_col: str = "normalized_mos",
) -> pd.DataFrame:
    path = _resolve_csv_path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if "image_path" not in df.columns:
        raise ValueError(
            f"{path} is missing required image_path column; found {df.columns.tolist()}"
        )

    if target_col not in df.columns:
        if target_col == "normalized_mos" and "mos" in df.columns:
            df = df.copy()
            df[target_col] = pd.to_numeric(df["mos"], errors="raise") / 100.0
        else:
            raise ValueError(
                f"{path} is missing target column {target_col!r}; "
                f"found {df.columns.tolist()}"
            )

    df = df.copy()
    df["image_path"] = [
        _resolve_image_path(value, path.parent) for value in df["image_path"].astype(str)
    ]
    df[target_col] = pd.to_numeric(df[target_col], errors="raise").astype("float32")
    df = df.dropna(subset=["image_path", target_col]).reset_index(drop=True)
    return df


def _count_possible_pairs(scores: np.ndarray, min_gap: float) -> int:
    ordered = np.sort(scores.astype(np.float32))
    count = 0
    right = 0
    total = len(ordered)
    for left, score in enumerate(ordered):
        if right <= left:
            right = left + 1
        while right < total and ordered[right] < score + min_gap:
            right += 1
        count += total - right
    return int(count)


def _row_from_indices(
    df: pd.DataFrame,
    target_col: str,
    index_a: int,
    index_b: int,
) -> dict[str, Any]:
    score_a = float(df.at[index_a, target_col])
    score_b = float(df.at[index_b, target_col])
    sign = 1 if score_a > score_b else -1
    return {
        "image_path_a": str(df.at[index_a, "image_path"]),
        "image_path_b": str(df.at[index_b, "image_path"]),
        "score_a": score_a,
        "score_b": score_b,
        "sign": sign,
        "gap": abs(score_a - score_b),
    }


def _enumerate_pairs(
    df: pd.DataFrame,
    target_col: str,
    min_gap: float,
) -> list[dict[str, Any]]:
    scores = df[target_col].to_numpy(dtype=np.float32)
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    rows: list[dict[str, Any]] = []
    right = 0
    for left, score in enumerate(ordered_scores):
        if right <= left:
            right = left + 1
        while right < len(order) and ordered_scores[right] < score + min_gap:
            right += 1
        for high in range(right, len(order)):
            rows.append(
                _row_from_indices(
                    df=df,
                    target_col=target_col,
                    index_a=int(order[high]),
                    index_b=int(order[left]),
                )
            )
    return rows


def _sample_pairs(
    df: pd.DataFrame,
    target_col: str,
    max_pairs: int,
    min_gap: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    scores = df[target_col].to_numpy(dtype=np.float32)
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    max_attempts = max(10000, max_pairs * 200)

    for _ in range(max_attempts):
        if len(rows) >= max_pairs:
            break
        pos_a = int(rng.integers(0, len(order)))
        score_a = float(ordered_scores[pos_a])
        lower_count = int(np.searchsorted(ordered_scores, score_a - min_gap, side="right"))
        upper_start = int(np.searchsorted(ordered_scores, score_a + min_gap, side="left"))
        upper_count = len(order) - upper_start
        candidate_count = lower_count + upper_count
        if candidate_count <= 0:
            continue

        pick = int(rng.integers(0, candidate_count))
        if pick < lower_count:
            pos_b = int(rng.integers(0, lower_count))
        else:
            pos_b = int(rng.integers(upper_start, len(order)))

        index_a = int(order[pos_a])
        index_b = int(order[pos_b])
        key = tuple(sorted((index_a, index_b)))
        if index_a == index_b or key in seen:
            continue
        seen.add(key)
        rows.append(
            _row_from_indices(
                df=df,
                target_col=target_col,
                index_a=index_a,
                index_b=index_b,
            )
        )

    return rows


def make_flive_pairs(
    csv_path: str | Path,
    output_csv: str | Path,
    target_col: str = "normalized_mos",
    max_pairs: int = 10000,
    min_gap: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive.")
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative.")

    df = load_iqa_csv(csv_path, target_col=target_col)
    possible_pairs = _count_possible_pairs(
        scores=df[target_col].to_numpy(dtype=np.float32),
        min_gap=min_gap,
    )
    requested_pairs = min(max_pairs, possible_pairs)
    if requested_pairs == 0:
        rows: list[dict[str, Any]] = []
    elif possible_pairs <= max_pairs:
        rows = _enumerate_pairs(
            df=df,
            target_col=target_col,
            min_gap=min_gap,
        )
    else:
        rows = _sample_pairs(
            df=df,
            target_col=target_col,
            max_pairs=requested_pairs,
            min_gap=min_gap,
            seed=seed,
        )

    if len(rows) > max_pairs:
        rows = rows[:max_pairs]

    columns = ["image_path_a", "image_path_b", "score_a", "score_b", "sign", "gap"]
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)

    summary = {
        "source_csv": str(_resolve_csv_path(csv_path)),
        "output_csv": str(output_path),
        "source_rows": int(len(df)),
        "possible_pairs": int(possible_pairs),
        "pair_count": int(len(rows)),
        "requested_max_pairs": int(max_pairs),
        "min_gap": float(min_gap),
        "seed": int(seed),
        "not_enough_pairs": bool(len(rows) < max_pairs),
    }
    print(
        "Created FLIVE pair CSV: "
        f"{summary['output_csv']} ({summary['pair_count']} pairs)"
    )
    if summary["not_enough_pairs"]:
        print(
            "Requested "
            f"{max_pairs} pairs but only wrote {summary['pair_count']} valid pairs."
        )
    return summary


def _decode_resize_with_pad(path: tf.Tensor, image_size: int) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, image_size, image_size)
    image.set_shape([image_size, image_size, 3])
    return image


def make_regression_dataset(
    csv_path: str | Path,
    target_col: str = "normalized_mos",
    image_size: int = 384,
    batch_size: int = 4,
    shuffle: bool = False,
    seed: int = 42,
    cache: bool = False,
    prefetch: bool = True,
) -> tf.data.Dataset:
    return make_arp_iqa_dataset(
        csv_path=csv_path,
        target_col=target_col,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        cache=cache,
        prefetch=prefetch,
    )


def make_pair_dataset(
    pair_csv: str | Path,
    image_size: int = 384,
    batch_size: int = 4,
    shuffle: bool = False,
    seed: int = 42,
    cache: bool = False,
    prefetch: bool = True,
) -> tf.data.Dataset:
    path = _resolve_csv_path(pair_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Pair CSV not found: {path}")

    df = pd.read_csv(path)
    required = ["image_path_a", "image_path_b", "sign"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {df.columns.tolist()}")

    df = df.copy()
    df["image_path_a"] = [
        _resolve_image_path(value, path.parent) for value in df["image_path_a"].astype(str)
    ]
    df["image_path_b"] = [
        _resolve_image_path(value, path.parent) for value in df["image_path_b"].astype(str)
    ]
    signs = pd.to_numeric(df["sign"], errors="raise").astype("float32")
    if not signs.isin([-1.0, 1.0]).all():
        raise ValueError("Pair signs must be -1 or +1.")

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["image_path_a"].astype(str).tolist(),
            df["image_path_b"].astype(str).tolist(),
            signs.tolist(),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(
            min(len(df), 10000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _map_fn(
        path_a: tf.Tensor,
        path_b: tf.Tensor,
        sign: tf.Tensor,
    ) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        image_a = _decode_resize_with_pad(path_a, image_size=image_size)
        image_b = _decode_resize_with_pad(path_b, image_size=image_size)
        return (image_a, image_b), tf.reshape(tf.cast(sign, tf.float32), (1,))

    dataset = dataset.map(_map_fn, num_parallel_calls=AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(AUTOTUNE)
    return dataset


__all__ = [
    "load_iqa_csv",
    "make_flive_pairs",
    "make_pair_dataset",
    "make_regression_dataset",
]
