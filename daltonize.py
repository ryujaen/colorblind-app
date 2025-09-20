import cv2
import numpy as np

SUPPORTED_TYPES = {"protan", "deutan", "tritan"}

# 참고: 표준화된 단순 근사 행렬(경량 데모용)
# 필요 시 Daltonization(시뮬+보정) 전체 파이프라인으로 교체 가능
MATS = {
    "protan": np.array([[0.567, 0.433, 0.000],
                        [0.558, 0.442, 0.000],
                        [0.000, 0.242, 0.758]], dtype=np.float32),
    "deutan": np.array([[0.625, 0.375, 0.000],
                        [0.700, 0.300, 0.000],
                        [0.000, 0.300, 0.700]], dtype=np.float32),
    "tritan": np.array([[0.950, 0.050, 0.000],
                        [0.000, 0.433, 0.567],
                        [0.000, 0.475, 0.525]], dtype=np.float32),
}

def _apply_matrix(bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
    """BGR -> RGB 행렬 적용 -> BGR (uint8)"""
    # to float in [0,1]
    img = bgr.astype(np.float32) / 255.0
    # BGR -> RGB
    img_rgb = img[..., ::-1]
    h, w, _ = img_rgb.shape
    flat = img_rgb.reshape(-1, 3)
    out = flat @ M.T
    out = np.clip(out, 0.0, 1.0).reshape(h, w, 3)
    # RGB -> BGR
    out_bgr = (out[..., ::-1] * 255.0 + 0.5).astype(np.uint8)
    return out_bgr

def _boost_saturation(bgr: np.ndarray, sat_gain: float = 1.12, cont_gain: float = 1.05) -> np.ndarray:
    """
    경량 채도/대비 보정.
    - 과도한 채도 폭주 방지 위해 HSV 기반으로 살짝만 증폭.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s *= sat_gain
    v = (v - 127.5) * cont_gain + 127.5  # 간단 대비
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    hsv_boost = cv2.merge([h, s, v]).astype(np.uint8)
    out = cv2.cvtColor(hsv_boost, cv2.COLOR_HSV2BGR)
    return out

def correct_image(bgr: np.ndarray, ctype: str = "protan") -> np.ndarray:
    """
    색각 유형별 간단 보정(행렬) + 경량 채도/대비 강화.
    - 데모/증빙용으로 충분한 시각적 구분 개선을 제공.
    """
    if ctype not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported type: {ctype}")

    M = MATS[ctype]
    corr = _apply_matrix(bgr, M)
    corr = _boost_saturation(corr, sat_gain=1.10 if ctype != "tritan" else 1.06, cont_gain=1.05)
    return corr
