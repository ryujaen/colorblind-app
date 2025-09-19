# image_utils.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Iterable, Union

import numpy as np
import cv2
from PIL import Image

ArrayLike = Union[np.ndarray, Image.Image, str, Path]


# -------------------------
# 기본 로드/세이브 유틸
# -------------------------
def load_image(x: ArrayLike) -> np.ndarray:
    """파일 경로/PIL/ndarray 입력을 모두 RGB ndarray(H,W,3)로 변환."""
    if isinstance(x, (str, Path)):
        img = Image.open(x).convert("RGBA")
    elif isinstance(x, Image.Image):
        img = x.convert("RGBA")
    elif isinstance(x, np.ndarray):
        # ndarray면 채널 수에 따라 정리
        arr = x
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[2] == 3:
            img = Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")
        elif arr.shape[2] == 4:
            img = Image.fromarray(arr.astype(np.uint8), "RGBA")
        else:
            raise ValueError("Unsupported ndarray shape.")
    else:
        raise TypeError("Unsupported input type.")
    return np.array(img)[:, :, :4]  # RGBA ndarray


def save_image(arr: np.ndarray, path: Union[str, Path], format: str | None = None):
    """RGBA/RGB ndarray를 파일로 저장. 확장자에 따라 자동 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.shape[2] == 4:
        img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    else:
        img = Image.fromarray(arr.astype(np.uint8), "RGB")
    img.save(path, format=format)


# -------------------------
# 배경 강제(흰/검정 등 단색)
# -------------------------
def add_solid_background(x: ArrayLike, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """투명 PNG 등에 단색 배경을 깔아 RGB로 반환."""
    rgba = load_image(x)
    rgb, a = rgba[:, :, :3], rgba[:, :, 3:4] / 255.0
    bg = np.zeros_like(rgb) + np.array(color, dtype=np.uint8)
    out = (rgb * a + bg * (1.0 - a)).astype(np.uint8)
    return out  # RGB


def batch_add_background(src_dir: Union[str, Path],
                         dst_dir: Union[str, Path],
                         color: Tuple[int, int, int] = (255, 255, 255),
                         exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".webp")):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    for p in src_dir.rglob("*"):
        if p.suffix.lower() in exts:
            out = add_solid_background(p, color)
            rel = p.relative_to(src_dir)
            save_image(out, Path(dst_dir, rel).with_suffix(".jpg"))  # 통일 저장


# -------------------------
# 원형 마스크(원 밖을 회색/검정/흰 등으로)
# -------------------------
def apply_circle_mask(x: ArrayLike,
                      bg: Tuple[int, int, int] = (200, 200, 200),
                      margin: int = 20,
                      feather: int = 0) -> np.ndarray:
    """
    이미지 중앙에 원형 마스크를 만들고, 원 밖은 단색(bg)으로 채움.
    - bg: 원 밖 배경색 (R,G,B)
    - margin: 테두리 여백(px) -> 원 반지름을 조금 줄임
    - feather: 가장자리 부드러움(가우시안 블러 커널, 짝수면 자동 홀수화)
    반환: RGB ndarray
    """
    rgba = load_image(x)
    h, w = rgba.shape[:2]
    rgb = rgba[:, :, :3]

    # 원형 마스크
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = max(1, min(center) - margin)
    cv2.circle(mask, center, radius, 255, thickness=-1)

    if feather and feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    mask_f = (mask / 255.0)[:, :, None]
    bg_img = np.zeros_like(rgb) + np.array(bg, dtype=np.uint8)
    out = (rgb * mask_f + bg_img * (1.0 - mask_f)).astype(np.uint8)
    return out  # RGB


def apply_circle_mask_to_path(in_path: Union[str, Path],
                              out_path: Union[str, Path],
                              **kwargs):
    out = apply_circle_mask(in_path, **kwargs)
    save_image(out, out_path)


def batch_apply_circle_mask(src_dir: Union[str, Path],
                            dst_dir: Union[str, Path],
                            bg: Tuple[int, int, int] = (200, 200, 200),
                            margin: int = 20,
                            feather: int = 0,
                            exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".webp")):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    for p in src_dir.rglob("*"):
        if p.suffix.lower() in exts:
            rel = p.relative_to(src_dir)
            out_path = Path(dst_dir, rel).with_suffix(".png")
            out = apply_circle_mask(p, bg=bg, margin=margin, feather=feather)
            save_image(out, out_path)
