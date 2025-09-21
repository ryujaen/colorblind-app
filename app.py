# app.py — TrueColor (daltonize.correct_image 안정 래퍼 적용 최종본)
import streamlit as st
st.set_page_config(page_title="TrueColor", layout="wide")

import numpy as np
import cv2
from PIL import Image, ImageOps

from daltonize import correct_image
from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# ===== CSS 커서 스타일 =====
st.markdown(
    """
    <style>
    div[data-baseweb="select"] { cursor: default !important; }
    div[data-baseweb="select"] * { cursor: default !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== 사이드바 =====
st.sidebar.title("TrueColor")
st.sidebar.caption("색각 이상자를 위한 색상 보정 웹앱 (daltonize)")

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["protan", "deutan", "tritan"],
    format_func=lambda x: {"protan": "Protanopia", "deutan": "Deuteranopia", "tritan": "Tritanopia"}[x],
)
max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)
alpha = st.sidebar.slider("보정 강도 (α)", 0.0, 2.0, 1.0, step=0.1,
                          help="0.0은 원본 유지, 1.0은 기본 보정, 2.0은 보정을 두 배 적용")
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
        # EXIF 회전 보정 + RGB 고정
        uploaded_img = ImageOps.exif_transpose(Image.open(img_file)).convert("RGB")
with col_u2:
    st.subheader("② 사용 방법")
    st.markdown(
        "- 좌측에서 이미지를 업로드하세요.\n"
        "- 색각 유형, 해상도, 보정 강도를 사이드바에서 조정하세요.\n"
        "- 아래에서 원본/보정 결과를 나란히 비교할 수 있습니다."
    )

st.divider()
if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드해 주세요.")
    st.stop()

# ===== daltonize.correct_image 안정 래퍼 =====
def run_daltonize_bgr(img_bgr: np.ndarray, ctype: str, alpha_val: float) -> np.ndarray:
    """
    입력: BGR uint8 [0..255]
    내부: RGB float [0..1] 로 변환 후 correct_image 호출
    출력: BGR uint8 [0..255]
    - correct_image가 alpha를 지원하지 않으면 외부 블렌딩으로 강도 반영
    - correct_image 결과가 0..255/0..1 둘 중 무엇이든 안전 처리
    """
    # BGR → RGB(float 0..1)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 1) 라이브러리에 alpha 함께 전달 시도
    out_rgb = None
    alpha_supported = True
    try:
        out_rgb = correct_image(rgb, ctype=ctype, alpha=alpha_val)
    except TypeError:
        alpha_supported = False
        out_rgb = correct_image(rgb, ctype=ctype)

    # 2) 결과 정규화 (0..1 로)
    if not isinstance(out_rgb, np.ndarray):
        # 실패 시 원본 그대로 반환
        return img_bgr

    o = out_rgb.astype(np.float32)
    if o.max() > 1.01:  # 0..255로 나온 경우
        o = o / 255.0
    o = np.clip(o, 0.0, 1.0)

    # 3) 라이브러리가 alpha 미지원이면 외부 블렌딩으로 강도 반영
    if not alpha_supported:
        o = (1.0 - alpha_val) * rgb + alpha_val * o
        o = np.clip(o, 0.0, 1.0)

    # 4) RGB(float) → BGR(uint8)
    out_bgr = cv2.cvtColor((o * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr

# ===== 처리 파이프라인 =====
pil_small = safe_resize(uploaded_img, target_long=max_width)
cv_small = pil_to_cv(pil_small)

corrected = run_daltonize_bgr(cv_small, ctype=ctype, alpha_val=alpha)

# 보정 차이(한 번만 표시)
diff = float(np.mean(np.abs(corrected.astype(np.int16) - cv_small.astype(np.int16))))
st.sidebar.write("보정 차이:", round(diff, 3))

# ===== 출력 =====
c1, c2 = st.columns([1, 1], gap="medium")
with c1:
    st.subheader("원본")
    st.image(cv_to_pil(cv_small), use_column_width=True)
with c2:
    st.subheader("보정 결과")
    st.image(cv_to_pil(corrected), use_column_width=True)

# 전/후 비교
st.subheader("전/후 비교 (가로 병치)")
compare_cv = side_by_side(cv_small, corrected, gap=16)
st.image(cv_to_pil(compare_cv), use_column_width=True)

st.caption("Tip: α(보정 강도)를 0.8~1.2에서 미세 조정해 보세요. 과하면 0.6~1.0으로 낮추면 자연스러워집니다.")
