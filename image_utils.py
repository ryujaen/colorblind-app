# image_utils.py
from __future__ import annotations
import io
from typing import Tuple
import numpy as np
from PIL import Image
import cv2

# --------- PIL <-> OpenCV 변환 ---------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL.Image -> OpenCV BGR ndarray (uint8). RGBA도 안전하게 처리."""
    if not isinstance(img, Image.Image):
        raise TypeError("pil_to_cv expects PIL.Image")
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGBA")
        arr = np.array(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        return bgr
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV BGR/BGRA ndarray -> PIL.Image"""
    if arr.ndim == 2:
        return Image.fromarray(arr)
    if arr.shape[2] == 4:
        rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# --------- 안전 리사이즈 ---------
def safe_resize(img: Image.Image | np.ndarray, max_side: int = 1200) -> Image.Image:
    """
    긴 변이 max_side를 넘으면 비율 유지 축소. PIL 이미지를 반환.
    """
    if isinstance(img, np.ndarray):
        img = cv_to_pil(img)
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)

# --------- 원형 마스크(바깥은 회색 처리) ---------
def apply_circle_mask(img: Image.Image | np.ndarray, outside_gray: int = 200, margin: int = 20) -> Image.Image:
    """
    중앙 원형 영역만 원본을 보이고, 바깥은 회색(기본 200)으로 처리.
    """
    pil = img if isinstance(img, Image.Image) else cv_to_pil(img)
    pil = pil.convert("RGB")
    arr = np.array(pil)
    h, w = arr.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = max(1, min(cx, cy) - margin)
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

    result = np.full_like(arr, outside_gray, dtype=np.uint8)
    result[mask == 255] = arr[mask == 255]
    return Image.fromarray(result)

# --------- 비교 이미지 합치기 ---------
def side_by_side(left: Image.Image | np.ndarray, right: Image.Image | np.ndarray, gap: int = 16) -> Image.Image:
    """
    좌우 이미지를 같은 높이로 맞춘 뒤 가로로 이어 붙인다.
    """
    L = left if isinstance(left, Image.Image) else cv_to_pil(left)
    R = right if isinstance(right, Image.Image) else cv_to_pil(right)
    L, R = L.convert("RGB"), R.convert("RGB")

    h = max(L.height, R.height)
    L = L.resize((int(L.width * h / L.height), h), Image.LANCZOS)
    R = R.resize((int(R.width * h / R.height), h), Image.LANCZOS)

    out = Image.new("RGB", (L.width + gap + R.width, h), (255, 255, 255))
    out.paste(L, (0, 0))
    out.paste(R, (L.width + gap, 0))
    return out
