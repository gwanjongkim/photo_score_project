# ICAA17K 공식 DAT 모델의 최소 스모크 테스트 스크립트
from pathlib import Path
import sys

sys.dont_write_bytecode = True

DAT_SMOKE_KWARGS = {
    "num_classes": 1,
    # Smoke-only DAT config: keeps the official 1024-wide head input, with one block per stage.
    "dim_stem": 128,
    "dims": [128, 256, 512, 1024],
    "depths": [1, 1, 1, 1],
    "stage_spec": [["L"], ["L"], ["D"], ["D"]],
    "heads": [4, 8, 16, 32],
    "groups": [-1, -1, 4, 8],
    "use_pes": [False, False, True, True],
    "strides": [-1, -1, 1, 1],
    "sr_ratios": [-1, -1, -1, -1],
    "offset_range_factor": [-1, -1, 2, 2],
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_official_repo_to_path(root: Path) -> Path:
    official_repo = root / "external" / "icaa_official_repo" / "ICAA17K_code"
    if not official_repo.is_dir():
        raise RuntimeError(f"Official ICAA17K repo path not found: {official_repo}")

    sys.path.insert(0, str(official_repo))
    return official_repo


def import_dat():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Failed to import torch. Install/activate the expected PyTorch environment.") from exc

    try:
        from models.dat import DAT
    except Exception as exc:
        raise RuntimeError(
            "Failed to import DAT from external/icaa_official_repo/ICAA17K_code/models/dat.py. "
            "Check that official repo dependencies such as timm and einops are installed."
        ) from exc

    return torch, DAT


def first_tensor_shape(value):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        for item in value:
            shape = first_tensor_shape(item)
            if shape is not None:
                return shape
    return None


def print_output_summary(outputs) -> None:
    print(f"outputs type: {type(outputs).__name__}")
    if isinstance(outputs, tuple):
        print(f"outputs length: {len(outputs)}")
        for idx, item in enumerate(outputs):
            if hasattr(item, "shape"):
                print(f"outputs[{idx}] type: {type(item).__name__}, shape: {tuple(item.shape)}")
            elif isinstance(item, (list, tuple)):
                print(
                    f"outputs[{idx}] type: {type(item).__name__}, "
                    f"length: {len(item)}, first tensor shape: {first_tensor_shape(item)}"
                )
            else:
                print(f"outputs[{idx}] type: {type(item).__name__}, shape: None")


def main() -> None:
    root = project_root()
    official_repo = add_official_repo_to_path(root)
    torch, DAT = import_dat()
    print("DAT import: ok")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"project root: {root}")
    print(f"official repo path: {official_repo}")
    print(f"device: {device}")

    try:
        model = DAT(**DAT_SMOKE_KWARGS).to(device)
        model.train()
    except Exception as exc:
        raise RuntimeError("DAT model construction failed with num_classes=1 and tiny smoke config.") from exc
    print("DAT construction: ok (num_classes=1)")

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    target = torch.tensor([[0.6, 0.7]], device=device)
    criterion = torch.nn.MSELoss()

    try:
        outputs = model(dummy_input)
    except Exception as exc:
        raise RuntimeError("DAT forward pass failed for dummy input shape (1, 3, 224, 224).") from exc

    print_output_summary(outputs)
    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise RuntimeError(f"Expected DAT forward to return a 3-item tuple, got {type(outputs).__name__}.")

    x, positions, references = outputs
    if tuple(x.shape) != (1, 2):
        raise RuntimeError(f"Expected x.shape == (1, 2), got {tuple(x.shape)}.")
    print(f"x shape assertion: ok ({tuple(x.shape)})")

    loss = criterion(x, target)
    print(f"loss: {loss.item():.6f}")

    try:
        loss.backward()
    except Exception as exc:
        raise RuntimeError("Backward pass failed for MSELoss against target shape (1, 2).") from exc

    print("backward: ok")
    print("ICAA17K DAT smoke test succeeded.")


if __name__ == "__main__":
    main()
