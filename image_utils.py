# image_utils.py (최종)

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
    """긴 변이 target_long을 넘으면 비율 유지 축소. PIL.Image 반환."""
    pil = img if isinstance(img, Image.Image) else cv_to_pil(img)
    w, h = pil.size
    m = max(w, h)
    if m <= target_long:
        return pil
    scale = target_long / float(m)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return pil.resize(new_size, Image.LANCZOS)

def apply_circle_mask(arr, bg_gray: int = 200, margin: int = 20):
    """
    입력/출력: OpenCV ndarray(BGR 또는 BGRA).
    중앙 원형만 원본, 바깥은 단색(bg_gray).
    """
    if not isinstance(arr, np.ndarray):
        arr = pil_to_cv(arr)

    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = max(1, min(cx, cy) - margin)
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

    ch = 4 if arr.ndim == 3 and arr.shape[2] == 4 else 3
    if ch == 3:
        bg = np.full((h, w, 3), bg_gray, dtype=np.uint8)
        out = bg.copy()
        out[mask == 255] = arr[mask == 255]
        return out
    else:
        bg = np.full((h, w, 4), (bg_gray, bg_gray, bg_gray, 255), dtype=np.uint8)
        out = bg.copy()
        out[mask == 255] = arr[mask == 255]
        return out
