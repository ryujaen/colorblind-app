# image_utils.py (FINAL)
from __future__ import annotations
import numpy as np
from PIL import Image
import cv2

def pil_to_cv(img: Image.Image) -> np.ndarray:
    if not isinstance(img, Image.Image):
        raise TypeError("pil_to_cv expects PIL.Image")
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGBA")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    if arr.shape[2] == 4:
        rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def safe_resize(img: Image.Image | np.ndarray, target_long: int = 1200) -> Image.Image:
    pil = img if isinstance(img, Image.Image) else cv_to_pil(img)
    w, h = pil.size
    m = max(w, h)
    if m <= target_long:
        return pil
    scale = target_long / float(m)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return pil.resize(new_size, Image.LANCZOS)

def apply_circle_mask(arr: np.ndarray, bg_gray: int = 200, margin: int = 20) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = pil_to_cv(arr)
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = max(1, min(cx, cy) - margin)
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)
    ch = 4 if arr.ndim == 3 and arr.shape[2] == 4 else 3
    if ch == 3:
        bg = np.full((h, w, 3), bg_gray, np.uint8)
        out = bg.copy()
        out[mask == 255] = arr[mask == 255]
        return out
    else:
        bg = np.full((h, w, 4), (bg_gray, bg_gray, bg_gray, 255), np.uint8)
        out = bg.copy()
        out[mask == 255] = arr[mask == 255]
        return out

def side_by_side(left: np.ndarray | Image.Image,
                 right: np.ndarray | Image.Image,
                 gap: int = 0) -> np.ndarray:   # 기본 gap도 0으로 해서 선 없음
    """
    좌우 이미지를 같은 높이로 맞춘 뒤 가로로 이어 붙인다.
    gap: 이미지 사이 간격 (기본 0)
    """
    L = left if isinstance(left, np.ndarray) else pil_to_cv(left)
    R = right if isinstance(right, np.ndarray) else pil_to_cv(right)

    # 같은 높이로 리사이즈
    h = max(L.shape[0], R.shape[0])
    def _resize_h(a: np.ndarray, h: int) -> np.ndarray:
        scale = h / a.shape[0]
        w = max(1, int(a.shape[1] * scale))
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LANCZOS4)

    Lr, Rr = _resize_h(L, h), _resize_h(R, h)

    # 새 캔버스 배경을 흰색 대신 검은색/투명 대신 "왼쪽 이미지 픽셀"로 채우기
    out = np.zeros((h, Lr.shape[1] + gap + Rr.shape[1], 3), np.uint8)
    out[:, :Lr.shape[1]] = Lr[:, :, :3]
    out[:, Lr.shape[1] + gap:] = Rr[:, :, :3]
    return out

