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
                 gap: int = 0,
                 bg_value: int = 0) -> np.ndarray:
    """
    좌우 이미지를 같은 높이로 맞춘 뒤 가로로 이어 붙인다.
    - 어떤 입력이 와도 최종 3채널(BGR)로 정규화
    - gap: 두 이미지 사이 간격 (px), 배경은 bg_value로 채움
    """
    # 1) numpy로 확보
    L = left  if isinstance(left,  np.ndarray) else pil_to_cv(left)
    R = right if isinstance(right, np.ndarray) else pil_to_cv(right)

    # 2) 채널 정규화 → 항상 3채널
    if L.ndim == 2:  # GRAY
        L = cv2.cvtColor(L, cv2.COLOR_GRAY2BGR)
    if R.ndim == 2:
        R = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
    if L.shape[2] == 4:  # BGRA -> BGR
        L = L[:, :, :3]
    if R.shape[2] == 4:
        R = R[:, :, :3]

    # 3) 높이 맞추기
    h = max(L.shape[0], R.shape[0])

    def resize_h(a: np.ndarray) -> np.ndarray:
        scale = h / a.shape[0]
        w = max(1, int(a.shape[1] * scale))
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LANCZOS4)

    Lr, Rr = resize_h(L), resize_h(R)

    # 4) 출력 캔버스 (항상 채널=3)
    out_w = Lr.shape[1] + gap + Rr.shape[1]
    out   = np.full((h, out_w, 3), bg_value, dtype=np.uint8)

    # 5) 복사
    out[:, :Lr.shape[1], :] = Lr
    out[:, Lr.shape[1] + gap:, :] = Rr
    return out