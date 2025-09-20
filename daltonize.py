# daltonize.py (최종본)
from __future__ import annotations
from typing import Any, Callable, Optional
import numpy as np
import cv2

# 앱에서 노출할 지원 타입
SUPPORTED_TYPES = ["protan", "deutan", "tritan"]


# -------------------------------------------------------------------------
# 내부 구현체 자동 탐색
# - 프로젝트마다 실제 보정 구현이 위치한 모듈명이 다를 수 있어
#   여기서 몇 가지 후보를 시도해보고, 있으면 그걸 사용
# - 없으면 None 반환(래퍼에서 안전히 처리)
# -------------------------------------------------------------------------
def _pick_impl() -> Optional[Callable[[np.ndarray, str], np.ndarray]]:
    candidates = [
        # 예시 후보들: 프로젝트 구조에 따라 알아서 맞는 게 있으면 import 성공
        "daltonize_impl",          # ex) 로컬 구현
        "dalton_impl",             # ex) 다른 이름으로 둔 구현
        "cvd_impl",                # ex) 색각 보정 구현 별도 모듈
        "truecolor.dalton_impl",   # ex) 패키지 내부 모듈
    ]
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["*"])
            if hasattr(mod, "correct_image") and callable(mod.correct_image):
                return getattr(mod, "correct_image")
        except Exception:
            continue
    return None


def _pick_simulator() -> Optional[Callable[[np.ndarray, str], np.ndarray]]:
    candidates = [
        "daltonize_impl",
        "dalton_impl",
        "cvd_impl",
        "truecolor.dalton_impl",
    ]
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["*"])
            if hasattr(mod, "simulate_cvd") and callable(mod.simulate_cvd):
                return getattr(mod, "simulate_cvd")
        except Exception:
            continue
    return None


# -------------------------------------------------------------------------
# 유틸: 입력/출력 안전성 보장
# -------------------------------------------------------------------------
def _ensure_bgr_uint8(arr: Any) -> np.ndarray:
    """이미지 배열을 BGR(uint8)로 정규화."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("correct_image expects a numpy.ndarray (OpenCV BGR)")
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("correct_image expects HxWxC with C in {3(BGR),4(BGRA)}")

    out = arr
    # BGRA -> BGR
    if out.shape[2] == 4:
        out = out[:, :, :3]

    # dtype 정규화
    if out.dtype != np.uint8:
        # float 등인 경우 0~1 또는 0~255일 수 있으므로 보수적으로 클램프
        out = np.clip(out, 0, 255).astype(np.uint8)

    return out


# -------------------------------------------------------------------------
# 공개 API: 보정 (alpha 강도 지원)
# -------------------------------------------------------------------------
def correct_image(arr: Any, ctype: str = "deutan", alpha: float = 1.0) -> np.ndarray:
    """
    색각 이상 보정 래퍼.
    - ctype: "protan" | "deutan" | "tritan"
    - alpha: 1.0 = 100% 보정, 0.0 = 원본, 그 사이 혼합(선형 보간)

    내부 구현체가 있으면 그걸 사용하고, 없으면 원본을 반환한다.
    """
    if ctype not in SUPPORTED_TYPES:
        # 알 수 없는 유형은 기본값으로 폴백
        ctype = "deutan"

    base = _ensure_bgr_uint8(arr)
    impl = _pick_impl()

    if impl is None:
        # 구현체가 없으면 보정 없이 alpha 무시하고 원본 반환(앱이 죽지 않게)
        return base

    try:
        corrected = impl(base, ctype)
        corrected = _ensure_bgr_uint8(corrected)
    except Exception:
        # 구현 호출 중 오류 시에도 앱이 죽지 않도록 원본 반환
        return base

    # alpha 혼합: out = alpha*corrected + (1-alpha)*original
    if alpha is None:
        alpha = 1.0
    alpha = float(alpha)
    if alpha == 1.0:
        return corrected
    if alpha == 0.0:
        return base

    # cv2.addWeighted은 자동 클램프 + 타입 보장
    out = cv2.addWeighted(corrected, alpha, base, 1.0 - alpha, 0.0)
    return out


# -------------------------------------------------------------------------
# 선택: 시뮬레이션(정상인이 특정 색각 이상처럼 보도록)
# -------------------------------------------------------------------------
def simulate_cvd(arr: Any, ctype: str = "deutan") -> np.ndarray:
    """
    색각 이상 시뮬레이션 래퍼(있으면 사용).
    내부 구현체가 없으면 보정 없이 원본 반환.
    """
    if ctype not in SUPPORTED_TYPES:
        ctype = "deutan"

    base = _ensure_bgr_uint8(arr)
    sim = _pick_simulator()
    if sim is None:
        return base

    try:
        out = sim(base, ctype)
        return _ensure_bgr_uint8(out)
    except Exception:
        return base
