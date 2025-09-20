import streamlit as st
st.set_page_config(page_title="TrueColor", layout="wide")

import numpy as np
from io import BytesIO
from PIL import Image, ImageOps

from daltonize import correct_image
from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# ===== CSS 커서 스타일 =====
st.markdown(
    """
    <style>
    div[data-baseweb="select"] {
        cursor: default !important;
    }
    div[data-baseweb="select"] * {
        cursor: default !important;
    }
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

# ===== 처리 파이프라인 =====
# 1) 안전 리사이즈(속도/메모리 절감) -> PIL
pil_small = safe_resize(uploaded_img, target_long=max_width)

# 2) OpenCV 배열로 변환 -> ndarray(BGR)
cv_small = pil_to_cv(pil_small)

# 3) 보정 강도(alpha) 슬라이더
alpha = st.sidebar.slider(
    "보정 강도 (alpha)", 0.0, 2.0, 1.0, step=0.1,
    help="1.0은 기본 보정, 2.0은 보정을 두 배 적용"
)

# 4) 보정 적용 -> ndarray(BGR)
corrected = correct_image(cv_small, ctype=ctype, alpha=alpha)

masked_src = cv_small
masked_dst = corrected

# ===== 출력 =====
# (1) 원본/보정 결과
c1, c2 = st.columns([1, 1], gap="medium")
with c1:
    st.subheader("원본")
    src_pil = cv_to_pil(masked_src)
    st.image(src_pil, use_column_width=True)
    buf_src = BytesIO()
    src_pil.save(buf_src, format="PNG")
    st.download_button(
        "원본 이미지 다운로드",
        data=buf_src.getvalue(),
        file_name=f"truecolor_original_{max_width}px.png",
        mime="image/png"
    )

with c2:
    st.subheader("보정 결과")
    dst_pil = cv_to_pil(masked_dst)
    st.image(dst_pil, use_column_width=True)
    buf_dst = BytesIO()
    dst_pil.save(buf_dst, format="PNG")
    st.download_button(
        "보정 이미지 다운로드",
        data=buf_dst.getvalue(),
        file_name=f"truecolor_{ctype}_alpha{alpha}_{max_width}px.png",
        mime="image/png"
    )

# (2) 전/후 비교
st.subheader("전/후 비교 (가로 병치)")
c3, c4 = st.columns([1, 1], gap="medium")
with c3:
    st.image(src_pil, use_column_width=True, caption="원본")
with c4:
    st.image(dst_pil, use_column_width=True, caption=f"보정 ({ctype}, α={alpha})")

st.caption("Tip: 사이드바에서 해상도와 보정 강도를 조절해 성능/품질/효과를 맞춰보세요.")
