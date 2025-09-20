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

def pil_to_cv(img: Image.Image) -> np.ndarray:
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGBA")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def safe_resize(img: Image.Image | np.ndarray, target_long: int = 1200) -> Image.Image:
    pil = img if isinstance(img, Image.Image) else pil_to_cv(img)
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

def side_by_side(left, right, gap: int = 0, bg_value: int = 0):
    import numpy as np, cv2
    from PIL import Image

    def _to_bgr(a):
        if isinstance(a, Image.Image):
            if a.mode in ("RGBA", "LA"):
                a = a.convert("RGB")
            a = cv2.cvtColor(np.array(a), cv2.COLOR_RGB2BGR)
            return a
        return a

    L = _to_bgr(left)
    R = _to_bgr(right)

    # 항상 3채널 보장
    if L.ndim == 2: L = cv2.cvtColor(L, cv2.COLOR_GRAY2BGR)
    if R.ndim == 2: R = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
    if L.shape[2] == 4: L = L[:, :, :3]
    if R.shape[2] == 4: R = R[:, :, :3]

    h = max(L.shape[0], R.shape[0])
    def rh(a):
        s = h / a.shape[0]
        w = max(1, int(a.shape[1] * s))
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LANCZOS4)

    Lr, Rr = rh(L), rh(R)
    out = np.full((h, Lr.shape[1] + gap + Rr.shape[1], 3), bg_value, np.uint8)  # ✅ 채널=3 고정
    out[:, :Lr.shape[1], :] = Lr
    out[:, Lr.shape[1] + gap:, :] = Rr
    return out
