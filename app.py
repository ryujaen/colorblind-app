# app.py — TrueColor (stable daltonize + fallback + cursor fix)
import streamlit as st
st.set_page_config(page_title="TrueColor", layout="wide")

import numpy as np
import cv2
from PIL import Image, ImageOps

from daltonize import correct_image
from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# ====== CSS: selectbox에서 텍스트 커서 숨기기 / 화살표 유지 ======
st.markdown(
    """
    <style>
    /* Streamlit selectbox는 내부에 input을 씁니다. */
    div[data-baseweb="select"] * { cursor: default !important; user-select: none !important; }
    /* 입력 캐럿(깜빡이) 숨김 */
    div[data-baseweb="select"] input { caret-color: transparent !important; }
    /* 드롭다운 열리는 영역(콤보박스)도 기본 화살표로 */
    div[data-baseweb="select"] [role="combobox"] { cursor: default !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====== 보정 래퍼 ======
def _srgb_to_float(rgb_uint8: np.ndarray) -> np.ndarray:
    return rgb_uint8.astype(np.float32) / 255.0

def _float_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _fallback_confusion_line(rgb_f: np.ndarray, key: str, alpha: float) -> np.ndarray:
    """간단 confusion-line 보정(확실히 효과를 보이게 하는 안전판) — RGB float in/out"""
    r, g, b = rgb_f[...,0], rgb_f[...,1], rgb_f[...,2]
    if key == "protan":
        r2, g2, b2 = r, g + 0.6*alpha*(r-g), b + 0.4*alpha*(r-b)
    elif key == "deutan":
        r2, g2, b2 = r + 0.6*alpha*(g-r), g, b + 0.4*alpha*(g-b)
    else:  # tritan
        r2, g2, b2 = r + 0.5*alpha*(b-r), g + 0.5*alpha*(b-g), b
    out = np.stack([r2, g2, b2], axis=-1)
    return _float_to_srgb(out)

def run_color_correction_bgr(img_bgr: np.ndarray, user_ctype: str, alpha: float) -> np.ndarray:
    """
    입력: BGR uint8 → 내부: RGB float → daltonize.correct_image 후보 여러 개 시도
    - 가장 변화 큰 결과 선택
    - 변화가 미미하면 confusion-line 폴백 적용
    출력: BGR uint8
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_f = _srgb_to_float(rgb)

    base = (user_ctype or "").lower()
    key = "protan" if base.startswith("prot") else ("deutan" if base.startswith("deut") else "tritan")
    # 라이브러리마다 토큰이 다를 수 있음 → 후보를 넓게 시도
    candidates = {
        "protan":  ["protanopia", "protan", "p"],
        "deutan":  ["deuteranopia", "deutan", "d"],
        "tritan":  ["tritanopia", "tritan", "t"],
    }[key]

    best = None
    best_diff = -1.0
    alpha_supported_used = False

    for token in candidates:
        # 1) alpha 지원 시도
        out = None
        alpha_supported = True
        try:
            out = correct_image(rgb_f, ctype=token, alpha=alpha)
        except TypeError:
            alpha_supported = False
            try:
                out = correct_image(rgb_f, ctype=token)
            except Exception:
                out = None
        except Exception:
            out = None

        if out is None:
            continue

        o = out.astype(np.float32)
        if o.max() > 1.01:  # 0..255로 온 케이스
            o = o / 255.0
        o = _float_to_srgb(o)

        # 2) alpha 미지원이면 외부에서 블렌딩
        if not alpha_supported:
            o = (1.0 - alpha) * rgb_f + alpha * o
            o = _float_to_srgb(o)

        # 3) 변화량 측정
        diff = float(np.mean(np.abs(o - rgb_f)))
        if diff > best_diff:
            best, best_diff, alpha_supported_used = o, diff, alpha_supported

        # 변화가 충분하면 조기 종료(빠르게 반응)
        if best_diff > 0.01:
            break

    # 후보 모두 실패/미미 → 폴백
    if best is None or best_diff < 1e-4:
        best = _fallback_confusion_line(rgb_f, key, alpha)

    out_bgr = cv2.cvtColor((best * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr

# ====== 사이드바 ======
st.sidebar.title("TrueColor")
st.sidebar.caption("색각 이상자를 위한 색상 보정 웹앱 (confusion-line + daltonize 래퍼)")

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["Protanopia", "Deuteranopia", "Tritanopia"],
    index=0,
)
# 내부 키로 정규화
ctype_key = "protan" if ctype.lower().startswith("prot") else ("deutan" if ctype.lower().startswith("deut") else "tritan")

alpha = st.sidebar.slider("보정 강도 (α)", 0.0, 2.0, 1.0, step=0.1)
max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)
st.sidebar.divider()

# ====== 본문 ======
st.title("TrueColor – 색상 보정 전/후 비교")

col_u1, col_u2 = st.columns(2)
uploaded_img = None
with col_u1:
    st.subheader("① 이미지/사진 입력")
    up = st.file_uploader("이미지 업로드 (JPG/PNG)", type=["jpg","jpeg","png"])
    if up:
        uploaded_img = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
with col_u2:
    st.subheader("② 사용 방법")
    st.markdown("- 이미지를 업로드하고 좌측에서 유형/강도를 조절하세요.\n- 아래에서 원본/보정을 비교할 수 있어요.")

st.divider()
if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드해 주세요.")
    st.stop()

# ====== 처리 파이프라인 ======
pil_small = safe_resize(uploaded_img, target_long=max_width)
cv_src = pil_to_cv(pil_small)  # BGR uint8

cv_dst = run_color_correction_bgr(cv_src, ctype_key, alpha)

# 보정 차이(한 번만 표시)
diff = float(np.mean(np.abs(cv_dst.astype(np.int16) - cv_src.astype(np.int16))))
st.sidebar.write("보정 차이:", round(diff, 3))

# ====== 출력 ======
c1, c2 = st.columns([1,1], gap="medium")
with c1:
    st.subheader("원본")
    st.image(cv_to_pil(cv_src), use_column_width=True)
with c2:
    st.subheader("보정 결과")
    st.image(cv_to_pil(cv_dst), use_column_width=True)

st.subheader("전/후 비교 (가로 병치)")
st.image(cv_to_pil(side_by_side(cv_src, cv_dst, gap=16)), use_column_width=True)
