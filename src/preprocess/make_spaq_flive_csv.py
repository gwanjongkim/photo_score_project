# SPAQ와 FLIVE 기술 화질 학습용 매니페스트를 생성하는 전처리 스크립트
import argparse
import tarfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def clean_relative_path(value):
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def manifest_columns(write_normalized):
    columns = ["image_path", "mos", "dataset", "is_patch", "relative_path"]
    if write_normalized:
        columns.append("normalized_mos")
    return columns


def require_columns(df, expected, source):
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required columns {missing}; "
            f"found {df.columns.tolist()}"
        )


def numeric_series(series, source, column):
    values = pd.to_numeric(series, errors="coerce")
    bad_count = int(values.isna().sum())
    if bad_count:
        raise ValueError(f"{source} has {bad_count} non-numeric values in {column}")
    return values.astype(float)


def image_path_for_layout(image_root, relative_path, archive_prefix):
    relative_path = clean_relative_path(relative_path)
    prefix = archive_prefix.strip("/")
    prefix_path = f"{prefix}/"
    if relative_path == prefix:
        no_prefix = ""
    elif relative_path.startswith(prefix_path):
        no_prefix = relative_path[len(prefix_path):]
    else:
        no_prefix = relative_path

    candidate_rel_paths = [relative_path]
    if no_prefix and no_prefix not in candidate_rel_paths:
        candidate_rel_paths.append(no_prefix)
    prefixed = f"{prefix}/{no_prefix}" if no_prefix else prefix
    if prefixed not in candidate_rel_paths:
        candidate_rel_paths.append(prefixed)

    candidates = [(image_root / Path(candidate)).resolve() for candidate in candidate_rel_paths]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate), True
    return str(candidates[0]), False


def attach_paths(df, image_root, archive_prefix):
    resolved = [
        image_path_for_layout(image_root, relative_path, archive_prefix)
        for relative_path in df["relative_path"]
    ]
    df = df.copy()
    df["image_path"] = [image_path for image_path, _ in resolved]
    df["_exists"] = [exists for _, exists in resolved]
    return df


def read_spaq_csv(csv_path, split_name):
    if not csv_path.exists():
        raise FileNotFoundError(f"SPAQ {split_name} CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    require_columns(
        df,
        ["image_id", "MOS", "Brightness", "Colorfulness", "Contrast", "Noisiness", "Sharpness"],
        str(csv_path),
    )
    return df


def build_spaq_manifest(source_df, image_root, write_normalized):
    source_mos = numeric_series(source_df["MOS"], "SPAQ CSV", "MOS")
    out = pd.DataFrame()
    out["relative_path"] = source_df["image_id"].map(clean_relative_path)
    out["mos"] = source_mos * 10.0
    out["dataset"] = "spaq"
    out["is_patch"] = 0
    if write_normalized:
        out["normalized_mos"] = source_mos / 10.0
    out = attach_paths(out, image_root, "spaq")
    return out[manifest_columns(write_normalized) + ["_exists"]]


def write_csv(df, path, write_normalized):
    path.parent.mkdir(parents=True, exist_ok=True)
    df[manifest_columns(write_normalized)].to_csv(path, index=False)


def mos_range(df):
    if df.empty:
        return "n/a"
    return f"{df['mos'].min():.6g} / {df['mos'].max():.6g}"


def print_path_counts(df, validate_paths):
    if not validate_paths:
        print("matched images: not validated")
        print("missing images: not validated")
        return
    matched = int(df["_exists"].sum())
    missing = int((~df["_exists"]).sum())
    print(f"matched images: {matched}")
    print(f"missing images: {missing}")


def build_spaq(args):
    train_csv = resolve_repo_path(args.spaq_train_csv)
    test_csv = resolve_repo_path(args.spaq_test_csv)
    image_root = resolve_repo_path(args.spaq_image_root)
    out_dir = resolve_repo_path(args.out_root) / "spaq"

    train_source = read_spaq_csv(train_csv, "train")
    test_source = read_spaq_csv(test_csv, "test")
    train_df = build_spaq_manifest(train_source, image_root, args.write_normalized)
    test_df = build_spaq_manifest(test_source, image_root, args.write_normalized)
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    all_path = out_dir / "labels_all.csv"
    write_csv(train_df, train_path, args.write_normalized)
    write_csv(test_df, test_path, args.write_normalized)
    write_csv(all_df, all_path, args.write_normalized)

    print("\n[SPAQ]")
    print(f"train rows: {len(train_df)}")
    print(f"test rows: {len(test_df)}")
    print_path_counts(all_df, args.validate_paths)
    print(f"min/max MOS: {mos_range(all_df)}")
    print("saved files:")
    print(train_path)
    print(test_path)
    print(all_path)
    return {
        "dataset": "spaq",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "matched": int(all_df["_exists"].sum()),
        "missing": int((~all_df["_exists"]).sum()),
        "mos_min": None if all_df.empty else float(all_df["mos"].min()),
        "mos_max": None if all_df.empty else float(all_df["mos"].max()),
        "paths": [train_path, test_path, all_path],
    }


def find_tar_member(tar, member_name):
    try:
        return tar.getmember(member_name)
    except KeyError:
        normalized_target = member_name.lstrip("./")
        for member in tar.getmembers():
            if member.name.lstrip("./") == normalized_target:
                return member
            if member.name.endswith(f"/{normalized_target}"):
                return member

    label_members = [
        member.name
        for member in tar.getmembers()
        if member.name.endswith(".csv") or "label" in member.name.lower()
    ]
    preview = label_members[:20]
    raise FileNotFoundError(
        f"Could not find {member_name} in tar archive. "
        f"CSV/label-like members found: {preview}"
    )


def read_flive_labels_from_tar(tar_path):
    if not tar_path.exists():
        raise FileNotFoundError(f"FLIVE tar not found: {tar_path}")
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            member = find_tar_member(tar, "flive/labels_image.csv")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"Could not read {member.name} from {tar_path}")
            with extracted:
                df = pd.read_csv(extracted)
    except tarfile.TarError as exc:
        raise RuntimeError(f"Could not read FLIVE labels from {tar_path}: {exc}") from exc

    require_columns(df, ["name", "mos"], f"{tar_path}:flive/labels_image.csv")
    return df


def build_flive_image_manifest(source_df, image_root, write_normalized):
    source_mos = numeric_series(source_df["mos"], "FLIVE labels_image.csv", "mos")
    out = pd.DataFrame()
    out["relative_path"] = source_df["name"].map(clean_relative_path)
    out["mos"] = source_mos
    out["dataset"] = "flive_image"
    out["is_patch"] = 0
    if write_normalized:
        out["normalized_mos"] = source_mos / 100.0
    out = attach_paths(out, image_root, "flive")
    return out[manifest_columns(write_normalized) + ["_exists"]]


def split_train_val_test(df, seed):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    train_df = shuffled.iloc[:train_end].reset_index(drop=True)
    val_df = shuffled.iloc[train_end:val_end].reset_index(drop=True)
    test_df = shuffled.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def build_flive(args):
    tar_path = resolve_repo_path(args.flive_tar)
    image_root = resolve_repo_path(args.flive_image_root)
    out_dir = resolve_repo_path(args.out_root) / "flive"

    source = read_flive_labels_from_tar(tar_path)
    all_df = build_flive_image_manifest(source, image_root, args.write_normalized)
    train_df, val_df, test_df = split_train_val_test(all_df, args.seed)

    all_path = out_dir / "labels_image_all.csv"
    train_path = out_dir / "image_train.csv"
    val_path = out_dir / "image_val.csv"
    test_path = out_dir / "image_test.csv"
    write_csv(all_df, all_path, args.write_normalized)
    write_csv(train_df, train_path, args.write_normalized)
    write_csv(val_df, val_path, args.write_normalized)
    write_csv(test_df, test_path, args.write_normalized)

    print("\n[FLIVE image-level]")
    print(f"total rows: {len(all_df)}")
    print(f"train rows: {len(train_df)}")
    print(f"val rows: {len(val_df)}")
    print(f"test rows: {len(test_df)}")
    print_path_counts(all_df, args.validate_paths)
    print(f"min/max MOS: {mos_range(all_df)}")
    print("saved files:")
    print(all_path)
    print(train_path)
    print(val_path)
    print(test_path)
    return {
        "dataset": "flive_image",
        "total_rows": len(all_df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "matched": int(all_df["_exists"].sum()),
        "missing": int((~all_df["_exists"]).sum()),
        "mos_min": None if all_df.empty else float(all_df["mos"].min()),
        "mos_max": None if all_df.empty else float(all_df["mos"].max()),
        "paths": [all_path, train_path, val_path, test_path],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SPAQ and FLIVE technical IQA manifest CSVs."
    )
    parser.add_argument("--dataset", choices=["spaq", "flive", "all"], default="all")
    parser.add_argument(
        "--spaq-train-csv",
        default="external/icaa_official_repo/SPAQ_code/dataset/SPAQ/spaqTrain.csv",
    )
    parser.add_argument(
        "--spaq-test-csv",
        default="external/icaa_official_repo/SPAQ_code/dataset/SPAQ/spaqTest.csv",
    )
    parser.add_argument("--spaq-image-root", default="data/raw/spaq")
    parser.add_argument("--flive-tar", default="/home/omen_pc1/dataset_zips/hf/flive.tar")
    parser.add_argument("--flive-image-root", default="data/raw/flive")
    parser.add_argument("--out-root", default="data/processed")
    parser.add_argument("--validate-paths", action="store_true")
    parser.add_argument("--write-normalized", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset in {"spaq", "all"}:
        build_spaq(args)
    if args.dataset in {"flive", "all"}:
        build_flive(args)


if __name__ == "__main__":
    main()
