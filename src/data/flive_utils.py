from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import re

from PIL import Image


PATCH_NAME_RE = re.compile(r"^(?P<base>.+?)_patch_(?P<idx>\d+)\.jpg$", re.IGNORECASE)


def parse_patch_filename(patch_rel_path: str) -> Tuple[str, int]:
    """
    예:
      patches/AVA__16945.jpg_patch_1.jpg
      -> ("AVA__16945.jpg", 1)
    """
    name = Path(patch_rel_path).name
    m = PATCH_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Invalid patch filename format: {patch_rel_path}")
    base_name = m.group("base")
    patch_idx = int(m.group("idx"))
    return base_name, patch_idx


def find_source_image(base_name: str, images_root: Path) -> Optional[Path]:
    """
    images_root 아래에서 base_name에 해당하는 원본 이미지를 찾는다.
    FLIVE 구조 예:
      images_root/
        EE371R/
        voc_emotic_ava/
        blur_dataset/
    """
    matches = list(images_root.rglob(base_name))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    # 같은 이름이 여러 개면 상대경로 길이가 짧은 것을 우선
    matches = sorted(matches, key=lambda p: len(str(p)))
    return matches[0]


def compute_patch_box(
    width: int,
    height: int,
    patch_idx: int,
    patch_size_ratio: float = 0.5,
) -> Tuple[int, int, int, int]:
    """
    현재 메타에 좌표 정보가 없으므로 deterministic crop을 사용한다.

    patch_idx:
      1 -> center crop
      2 -> left/top bias crop
      3 -> right/bottom bias crop
      4 -> top-left
      5 -> top-right
      6 -> bottom-left
      7 -> bottom-right

    그 이상은 modulo로 순환.

    patch_size_ratio=0.5이면 원본 짧은 변의 50% 정사각형 crop.
    """
    short_side = min(width, height)
    crop_size = max(32, int(short_side * patch_size_ratio))

    # clamp
    crop_size = min(crop_size, width, height)

    positions = [
        ("center", 0.5, 0.5),
        ("left_topbias", 0.25, 0.5 if width < height else 0.25),
        ("right_bottombias", 0.75, 0.5 if width < height else 0.75),
        ("top_left", 0.25, 0.25),
        ("top_right", 0.75, 0.25),
        ("bottom_left", 0.25, 0.75),
        ("bottom_right", 0.75, 0.75),
    ]

    _, px, py = positions[(patch_idx - 1) % len(positions)]

    cx = int(width * px)
    cy = int(height * py)

    left = max(0, min(width - crop_size, cx - crop_size // 2))
    top = max(0, min(height - crop_size, cy - crop_size // 2))
    right = left + crop_size
    bottom = top + crop_size

    return left, top, right, bottom


def make_patch_image(
    src_path: Path,
    dst_path: Path,
    patch_idx: int,
    patch_size_ratio: float = 0.5,
    resize_to: Optional[int] = 224,
    quality: int = 95,
) -> None:
    """
    원본 이미지에서 deterministic crop을 생성하고 저장한다.
    이미 dst_path가 있으면 호출자가 skip 처리해야 한다.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        box = compute_patch_box(w, h, patch_idx, patch_size_ratio=patch_size_ratio)
        patch = img.crop(box)

        if resize_to is not None:
            patch = patch.resize((resize_to, resize_to), Image.Resampling.BILINEAR)

        patch.save(dst_path, format="JPEG", quality=quality)
