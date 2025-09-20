from __future__ import annotations
import numpy as np
import cv2
from PIL import Image

# ---------- PIL <-> OpenCV ----------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL.Image -> OpenCV BGR/BGRA ndarray (uint8)."""
    if not isinstance(img, Image.Image):
        raise TypeError("pil_to_cv expects PIL.Image")
    # RGBA/LA도 안전 처리
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGBA")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV BGR/BGRA/GRAY ndarray -> PIL.Image"""
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
    긴 변이 target_long을 넘으면 비율 유지 축소. PIL 이미지를 반환.
    """
    pil = img if isinstance(img, Image.Image) else cv_to_pil(img)
    w, h = pil.size
    m = max(w, h)
    if m <= target_long:
        return pil
    s = target_long / float(m)
    new_size = (max(1, int(w * s)), max(1, int(h * s)))
    return pil.resize(new_size, Image.LANCZOS)

# ---------- 가로 병치(안전판) ----------
def side_by_side(left: Image.Image | np.ndarray,
                 right: Image.Image | np.ndarray,
                 gap: int = 0,
                 bg_value: int = 0) -> np.ndarray:
    """
    좌우 이미지를 같은 높이로 맞춰 가로로 붙인다.
    어떤 입력이 와도 최종 3채널(BGR)로 정규화.
    """
    # numpy 배열로 확보
    L = left  if isinstance(left,  np.ndarray) else pil_to_cv(left)
    R = right if isinstance(right, np.ndarray) else pil_to_cv(right)

    # 채널 정규화: 항상 3채널
    if L.ndim == 2: L = cv2.cvtColor(L, cv2.COLOR_GRAY2BGR)
    if R.ndim == 2: R = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
    if L.shape[2] == 4: L = L[:, :, :3]
    if R.shape[2] == 4: R = R[:, :, :3]

    # 높이 맞추기
    h = max(L.shape[0], R.shape[0])
    def _resize_h(a: np.ndarray) -> np.ndarray:
        scale = h / a.shape[0]
        w = max(1, int(a.shape[1] * scale))
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LANCZOS4)
    Lr, Rr = _resize_h(L), _resize_h(R)

    # 출력 캔버스: 채널=3 고정
    out_w = Lr.shape[1] + gap + Rr.shape[1]
    out   = np.full((h, out_w, 3), bg_value, dtype=np.uint8)

    # 복사
    out[:, :Lr.shape[1], :] = Lr
    out[:, Lr.shape[1] + gap:, :] = Rr
    return out
