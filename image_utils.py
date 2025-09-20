# image_utils.py
from __future__ import annotations
import numpy as np
from PIL import Image
import cv2
from PIL import Image, ImageDraw


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
def apply_circle_mask(arr: np.ndarray, margin: int = 20) -> np.ndarray:
    """
    입력: OpenCV ndarray(BGR)
    출력: OpenCV ndarray(BGRA) - 원형 영역만 원본, 바깥은 투명
    """
    if not isinstance(arr, np.ndarray):
        arr = pil_to_cv(arr)

    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = w // 2, h // 2
    r = max(1, min(cx, cy) - margin)

    # 원형 마스크 (흰색 부분 = 보이는 영역)
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

    # BGR → BGRA (알파 채널 추가)
    if arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)

    # 바깥은 알파 0, 안쪽은 알파 255
    arr[:, :, 3] = mask

    return arr


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

def make_circular_rgba(pil_img: Image.Image, margin: int = 0) -> Image.Image:
    """
    Ishihara plate처럼 원형 콘텐츠만 보이도록, 원 밖은 투명 처리한 RGBA 이미지 반환
    """
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")

    w, h = pil_img.size
    r = min(w, h) // 2 - max(0, margin)
    cx, cy = w // 2, h // 2

    # 원형 마스크 만들기
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    # 알파 채널 적용
    r_, g_, b_, _ = pil_img.split()
    out = Image.merge("RGBA", (r_, g_, b_, mask))
    return out