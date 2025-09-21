# app.py (정리/보강 최종본)
import numpy as np
import cv2
from io import BytesIO
import streamlit as st
st.set_page_config(page_title="TrueColor", layout="wide")

from PIL import Image, ImageOps

from daltonize import correct_image
from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# ===== CSS (selectbox 위에 항상 화살표 커서) =====
st.markdown(
    """
    <style>
    div[data-baseweb="select"] { cursor: default !important; }
    div[data-baseweb="select"] * { cursor: default !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== 사이드바 =====
st.sidebar.title("TrueColor")
st.sidebar.caption("색각 이상자를 위한 색상 보정 웹앱 (요약 데모)")

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["protan", "deutan", "tritan"],
    format_func=lambda x: {"protan": "Protanopia", "deutan": "Deuteranopia", "tritan": "Tritanopia"}[x],
)

max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)

alpha = st.sidebar.slider(
    "보정 강도 (α)", 0.0, 2.0, 1.0, step=0.1,
    help="0.0은 원본 유지, 1.0은 기본 보정, 2.0은 보정을 두 배 적용"
)

st.sidebar.divider()

# ===== 본문 =====
st.title("TrueColor – 색상 보정 전/후 비교")
st.write("**이미지 업로드 → 보정 적용 → 전/후 비교**")

col_u1, col_u2 = st.columns(2)
uploaded_img = None

with col_u1:
    st.subheader("① 이미지/사진 입력")
    img_file = st.file_uploader("이미지 업로드 (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if img_file:
        # EXIF 회전 보정 + RGB
        uploaded_img = ImageOps.exif_transpose(Image.open(img_file)).convert("RGB")

with col_u2:
    st.subheader("② 사용 방법")
    st.markdown(
        "- 좌측에서 이미지를 업로드하세요.\n"
        "- 색각 유형, 해상도, 보정 강도를 사이드바에서 조정하세요.\n"
        "- 아래에서 원본/보정 결과를 나란히 비교하고 다운로드할 수 있습니다."
    )

st.divider()

if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드해 주세요.")
    st.stop()

# ===== 처리 파이프라인 =====
# 1) 안전 리사이즈(속도/메모리 절감) -> PIL
pil_small = safe_resize(uploaded_img, target_long=max_width)

# 2) OpenCV 배열로 변환 -> ndarray(BGR)
cv_small = pil_to_cv(pil_small)

# ===== 보정 테스트 =====
rgb = cv2.cvtColor(cv_small, cv2.COLOR_BGR2RGB)

try:
    base_rgb = correct_image(rgb, ctype=ctype_norm)
except TypeError:
    base_rgb = correct_image(rgb, ctype=ctype_norm)

# 결과를 다시 BGR로 변환
if isinstance(base_rgb, np.ndarray):
    base = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
else:
    base = cv_small.copy()

# 보정 강도 반영
corrected = (
    cv_small.astype(np.float32) * (1.0 - alpha) +
    base.astype(np.float32)      * alpha
).clip(0, 255).astype("uint8")

# 디버깅용 차이 출력
diff = np.mean(np.abs(corrected.astype(np.int16) - cv_small.astype(np.int16)))
st.sidebar.write("보정 차이:", diff)

def normalize_ctype(c: str) -> str:
    c = (c or "").lower()
    mapping = {
        "protan": "protanopia", "protanopia": "protanopia",
        "deutan": "deuteranopia", "deuteranopia": "deuteranopia",
        "tritan": "tritanopia", "tritanopia": "tritanopia",
    }
    return mapping.get(c, c)

ctype_norm = normalize_ctype(ctype)

def run_correct(img_bgr: np.ndarray, ctype_str: str, alpha_val: float) -> np.ndarray:
    """
    correct_image가 어떤 색상공간/시그니처를 기대하는지 모를 때 안전하게 처리:
    1) BGR 그대로 넣어본 뒤 변화 없으면
    2) RGB로 변환해 넣고, 결과를 다시 BGR로 복원
    또한 alpha 인자 미지원이면 자동 호환.
    """
    # 1) BGR 그대로 시도
    try:
        try:
            out1 = correct_image(img_bgr, ctype=ctype_str, alpha=alpha_val)
        except TypeError:
            out1 = correct_image(img_bgr, ctype=ctype_str)
    except Exception:
        out1 = None

    # out1이 실패했거나, 결과가 원본과 사실상 동일하면 RGB 시도
    need_rgb_try = (
        out1 is None or
        (isinstance(out1, np.ndarray) and out1.shape == img_bgr.shape and
         np.mean(np.abs(out1.astype(np.int16) - img_bgr.astype(np.int16))) < 0.5)
    )

    if need_rgb_try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            try:
                out2_rgb = correct_image(rgb, ctype=ctype_str, alpha=alpha_val)
            except TypeError:
                out2_rgb = correct_image(rgb, ctype=ctype_str)
            if isinstance(out2_rgb, np.ndarray) and out2_rgb.ndim >= 2:
                out2_bgr = cv2.cvtColor(out2_rgb, cv2.COLOR_RGB2BGR)
                return out2_bgr
        except Exception:
            pass

    # 여기까지 왔으면 out1을 신뢰
    return out1 if isinstance(out1, np.ndarray) else img_bgr

# 실제 적용
base = run_correct(cv_small, ctype_norm, alpha)

# alpha 블렌딩(강도 체감)
corrected = (
    cv_small.astype(np.float32) * (1.0 - alpha) +
    base.astype(np.float32)      * alpha
).clip(0, 255).astype("uint8")