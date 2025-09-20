# image_utils.py
from __future__ import annotations
import numpy as np
from PIL import Image
import cv2

# ---------- PIL <-> OpenCV ----------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL.Image -> OpenCV BGR ndarray (uint8)."""
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
    """OpenCV BGR/BGRA ndarray -> PIL.Image."""
    if arr.ndim == 2:
        return Image.fromarray(arr)
    if arr.shape[2] == 4:
        rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# ---------- 안전 리사이즈 ----------
def safe_resize(img: Image.Image | np.ndarray, target_long: int = 1200) -> Image.Image:
    """
    긴 변이 target_long을 넘으면 비율 유지 축소. 항상 PIL.Image 반환.
    """
    pil = img if isinstance(img, Image.Image) else cv_to_pil(img)
    w, h = pil.size
    m = max(w, h)
    if m <= target_long:
        return pil
    scale = target_long / float(m)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return pil.resize(new_size, Image.LANCZOS)

# ---------- 원형 마스크 ----------
def apply_circle_mask(arr: np.ndarray, bg_gray: int = 200, margin: int = 20) -> np.ndarray:
    """
    입력/출력: OpenCV ndarray(BGR 또는 BGRA).
    중앙 원형 부분만 원본 보이고, 바깥은 회색(bg_gray).
    """
    if not isinstance(arr, np.ndarray):
        arr = pil_to_cv(arr)  # 안전장치

    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = max(1, min(cx, cy) - margin)
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

    # 결과 캔버스: 채널 수 유지
    ch = 4 if arr.ndim == 3 and arr.shape[2] == 4 else 3
    if ch == 3:
        bg = np.full((h, w, 3), bg_gray, dtype=np.uint8)
        out = bg.copy()
        out[mask == 255] = arr[mask == 255]
        return out
    else:
        # BGRA
        bg = np.full((h, w, 4), (bg_gray, bg_gray, bg_gray, 255), dtype=np.uint8)
        out = bg.copy()
        out[mask == 255] = arr[mask == 255]
        return out

# ---------- 좌우 비교 합치기 ----------
def side_by_side(left: np.ndarray | Image.Image,
                 right: np.ndarray | Image.Image,
                 gap: int = 16) -> np.ndarray:
    """
    좌우 이미지를 같은 높이로 맞춘 뒤 가로로 이어붙임.
    입력/출력: ndarray(BGR).
    """
    L = left if isinstance(left, np.ndarray) else pil_to_cv(left)
    R = right if isinstance(right, np.ndarray) else pil_to_cv(right)

    # 높이 맞추기
    h = max(L.shape[0], R.shape[0])
    def _resize_h(a: np.ndarray, h: int) -> np.ndarray:
        scale = h / a.shape[0]
        w = max(1, int(a.shape[1] * scale))
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LANCZOS4)

    Lr = _resize_h(L, h)
    Rr = _resize_h(R, h)

    # 캔버스 생성 (BGR)
    out = np.full((h, Lr.shape[1] + gap + Rr.shape[1], 3), 255, dtype=np.uint8)
    out[:, :Lr.shape[1]] = Lr[:, :, :3]
    out[:, Lr.shape[1] + gap:] = Rr[:, :, :3]
    return out
