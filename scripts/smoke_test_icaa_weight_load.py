# ICAA17K 공식 DAT 체크포인트 로딩을 확인하는 최소 스모크 테스트
from pathlib import Path
from pprint import pformat
import sys

sys.dont_write_bytecode = True

CHECKPOINT_REL_PATH = "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"

OFFICIAL_DAT_BASE_CONFIG = {
    "img_size": 224,
    "patch_size": 4,
    "num_classes": 1,
    "expansion": 4,
    "dim_stem": 128,
    "dims": [128, 256, 512, 1024],
    "depths": [2, 2, 18, 2],
    "stage_spec": [
        ["L", "S"],
        ["L", "S"],
        ["L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D"],
        ["L", "D"],
    ],
    "heads": [4, 8, 16, 32],
    "window_sizes": [7, 7, 7, 7],
    "groups": [-1, -1, 4, 8],
    "use_pes": [False, False, True, True],
    "dwc_pes": [False, False, False, False],
    "strides": [-1, -1, 1, 1],
    "sr_ratios": [-1, -1, -1, -1],
    "offset_range_factor": [-1, -1, 2, 2],
    "no_offs": [False, False, False, False],
    "fixed_pes": [False, False, False, False],
    "use_dwc_mlps": [False, False, False, False],
    "use_conv_patches": False,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.5,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_official_repo_to_path(root: Path) -> Path:
    official_repo = root / "external" / "icaa_official_repo" / "ICAA17K_code"
    if not official_repo.is_dir():
        raise RuntimeError(f"Official ICAA17K repo path not found: {official_repo}")
    sys.path.insert(0, str(official_repo))
    return official_repo


def import_dat_and_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Failed to import torch.") from exc

    try:
        from models.dat import DAT
    except Exception as exc:
        raise RuntimeError(
            "Failed to import DAT from external/icaa_official_repo/ICAA17K_code/models/dat.py. "
            "Check that official repo dependencies such as timm and einops are installed."
        ) from exc

    return torch, DAT


def load_checkpoint(torch, checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def tensor_shape(value):
    return tuple(value.shape) if hasattr(value, "shape") else None


def print_checkpoint_summary(checkpoint) -> None:
    keys = list(checkpoint.keys())
    print(f"checkpoint key count: {len(keys)}")
    print("first 10 checkpoint keys and shapes:")
    for key in keys[:10]:
        print(f"  {key}: {tensor_shape(checkpoint[key])}")
    print(f"patch_proj.0.weight shape: {tensor_shape(checkpoint['patch_proj.0.weight'])}")


def infer_config_from_checkpoint(checkpoint):
    config = dict(OFFICIAL_DAT_BASE_CONFIG)
    patch_shape = tensor_shape(checkpoint.get("patch_proj.0.weight"))
    if patch_shape is None or len(patch_shape) != 4:
        raise RuntimeError("Checkpoint is missing tensor key patch_proj.0.weight.")

    dim_stem = patch_shape[0]
    if dim_stem != config["dim_stem"]:
        raise RuntimeError(f"Checkpoint dim_stem is {dim_stem}, but official base config expects {config['dim_stem']}.")
    return config


def check_state_dict_compatibility(model, checkpoint):
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    checkpoint_keys = set(checkpoint.keys())
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatches = []

    for key in sorted(model_keys & checkpoint_keys):
        if tensor_shape(model_state[key]) != tensor_shape(checkpoint[key]):
            shape_mismatches.append((key, tensor_shape(model_state[key]), tensor_shape(checkpoint[key])))

    return missing, unexpected, shape_mismatches


def main() -> None:
    root = project_root()
    official_repo = add_official_repo_to_path(root)
    torch, DAT = import_dat_and_torch()
    print("DAT import: ok")
    print(f"official repo path: {official_repo}")

    checkpoint_path = root / CHECKPOINT_REL_PATH
    checkpoint = load_checkpoint(torch, checkpoint_path)
    if not hasattr(checkpoint, "keys"):
        raise RuntimeError(f"Expected checkpoint to be a state_dict-like object, got {type(checkpoint).__name__}.")
    print_checkpoint_summary(checkpoint)

    config = infer_config_from_checkpoint(checkpoint)
    print("model config used:")
    print(pformat(config, sort_dicts=False))

    try:
        model = DAT(**config)
    except Exception as exc:
        raise RuntimeError("DAT model construction failed with official base config.") from exc

    missing, unexpected, shape_mismatches = check_state_dict_compatibility(model, checkpoint)
    print(f"missing keys: {missing}")
    print(f"unexpected keys: {unexpected}")
    if shape_mismatches:
        print("shape mismatches:")
        for key, model_shape, checkpoint_shape in shape_mismatches:
            print(f"  {key}: model={model_shape}, checkpoint={checkpoint_shape}")
        raise RuntimeError("Checkpoint shape mismatches found; refusing to load.")
    if missing or unexpected:
        raise RuntimeError("Missing or unexpected keys found; refusing to load.")

    load_result = model.load_state_dict(checkpoint, strict=True)
    print(f"load result: {load_result}")
    print("checkpoint load succeeded: yes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    print(f"device: {device}")

    with torch.no_grad():
        outputs = model(dummy_input)

    if not isinstance(outputs, tuple):
        raise RuntimeError(f"Expected DAT forward to return a tuple, got {type(outputs).__name__}.")
    print(f"output tuple length: {len(outputs)}")
    if len(outputs) != 3:
        raise RuntimeError(f"Expected DAT forward to return 3 values, got {len(outputs)}.")

    x, positions, references = outputs
    print(f"x shape: {tuple(x.shape)}")
    if tuple(x.shape) != (1, 2):
        raise RuntimeError(f"Expected x.shape == (1, 2), got {tuple(x.shape)}.")
    print(f"first output values: {[round(value, 6) for value in x[0].detach().cpu().tolist()]}")
    print("ICAA17K DAT checkpoint weight-load smoke test succeeded.")


if __name__ == "__main__":
    main()
