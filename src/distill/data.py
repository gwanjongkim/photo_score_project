from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.distill.preprocessing import decode_resize_normalize_student_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTOTUNE = tf.data.AUTOTUNE


def canonicalize_image_key(path_value: object) -> str | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return None
    try:
        return str(Path(text).expanduser().resolve(strict=False))
    except OSError:
        return text


def infer_split_name(manifest_path: Path) -> str:
    lowered = manifest_path.stem.lower()
    for split_name in ("train", "val", "test"):
        if split_name in lowered:
            return split_name
    return "unspecified"


def resolve_manifest_image_path(path_value: object, manifest_path: Path) -> Path:
    candidate = Path(str(path_value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)

    repo_candidate = (PROJECT_ROOT / candidate).resolve(strict=False)
    if repo_candidate.exists():
        return repo_candidate

    manifest_relative_candidate = (manifest_path.parent / candidate).resolve(strict=False)
    if manifest_relative_candidate.exists():
        return manifest_relative_candidate

    return repo_candidate


def load_source_manifests(csv_paths: list[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path_value in csv_paths:
        csv_path = Path(csv_path_value)
        if not csv_path.is_absolute():
            csv_path = (PROJECT_ROOT / csv_path).resolve(strict=False)
        frame = pd.read_csv(csv_path)
        if "image_path" not in frame.columns:
            raise ValueError(f"Manifest is missing image_path: {csv_path}")
        frame = frame.copy()
        inferred_split = infer_split_name(csv_path)
        if "split" not in frame.columns:
            frame["split"] = inferred_split
        else:
            frame["split"] = frame["split"].fillna("").astype(str).str.strip()
            unique_splits = sorted({value for value in frame["split"].tolist() if value})
            if unique_splits and len(unique_splits) == 1 and unique_splits[0] != inferred_split:
                print(
                    f"Split override: manifest {csv_path} declared split={unique_splits[0]!r}, "
                    f"using inferred split={inferred_split!r} from filename."
                )
                frame["split"] = inferred_split
            else:
                frame.loc[frame["split"] == "", "split"] = inferred_split
        frame["source_manifest"] = str(csv_path)
        frame["source_row_index"] = range(len(frame))
        frame["image_path"] = frame["image_path"].apply(lambda value: str(resolve_manifest_image_path(value, csv_path)))
        frames.append(frame)

    if not frames:
        raise ValueError("No input manifests were provided.")

    combined = pd.concat(frames, ignore_index=True)
    combined["__image_key"] = combined["image_path"].apply(canonicalize_image_key)
    combined = combined.dropna(subset=["__image_key"]).copy()
    combined = combined.drop_duplicates(subset=["__image_key", "split"], keep="first").reset_index(drop=True)
    return combined.drop(columns=["__image_key"])


def load_teacher_manifest(csv_path: str | Path, target_col: str = "teacher_aesthetic_score") -> pd.DataFrame:
    manifest_path = Path(csv_path)
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve(strict=False)
    frame = pd.read_csv(manifest_path)
    required_columns = {"image_path", "split", target_col}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Teacher manifest is missing required columns: {', '.join(missing_columns)}"
        )
    frame = frame.copy()
    frame["image_path"] = frame["image_path"].apply(lambda value: str(resolve_manifest_image_path(value, manifest_path)))
    frame["split"] = frame["split"].astype(str)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna(subset=[target_col]).reset_index(drop=True)
    return frame


def summarize_cross_split_duplicates(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty or "image_path" not in frame.columns or "split" not in frame.columns:
        return {
            "num_cross_split_duplicates": 0,
            "examples": [],
        }

    temp = frame[["image_path", "split"]].copy()
    temp["__image_key"] = temp["image_path"].apply(canonicalize_image_key)
    temp = temp.dropna(subset=["__image_key"]).copy()
    grouped = (
        temp.groupby("__image_key")["split"]
        .agg(lambda values: sorted({str(value) for value in values}))
        .reset_index(name="splits")
    )
    grouped["num_splits"] = grouped["splits"].apply(len)
    duplicates = grouped[grouped["num_splits"] > 1].copy()
    examples = duplicates.head(10).to_dict(orient="records")
    return {
        "num_cross_split_duplicates": int(len(duplicates)),
        "examples": examples,
    }


def select_splits(frame: pd.DataFrame, splits: list[str] | tuple[str, ...]) -> pd.DataFrame:
    split_values = {str(value) for value in splits}
    selected = frame[frame["split"].astype(str).isin(split_values)].copy()
    if selected.empty:
        raise ValueError(f"No rows matched splits: {sorted(split_values)}")
    return selected.reset_index(drop=True)


def decode_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    return decode_resize_normalize_student_image(path, image_size=image_size)


def make_student_dataset(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    target_col: str = "teacher_aesthetic_score",
    training: bool = False,
    shuffle_buffer: int = 4096,
    augment_horizontal_flip: bool = False,
    cache: bool = False,
) -> tf.data.Dataset:
    paths = frame["image_path"].astype(str).tolist()
    targets = frame[target_col].astype("float32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))

    def _map_fn(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = decode_image(path, image_size=image_size)
        if training and augment_horizontal_flip:
            image = tf.image.random_flip_left_right(image)
        return image, tf.cast(target, tf.float32)

    dataset = dataset.map(_map_fn, num_parallel_calls=AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(min(len(frame), int(shuffle_buffer)), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset
