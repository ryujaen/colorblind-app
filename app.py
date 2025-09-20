import streamlit as st
st.set_page_config(page_title="TrueColor", layout="wide")  # 반드시 첫 번째 Streamlit 호출(1회만)

import numpy as np
from PIL import Image

from daltonize import correct_image  # 보정 함수
from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# ===== 사이드바 =====
st.sidebar.title("TrueColor")
st.sidebar.caption("색각 이상자를 위한 색상 보정 웹앱 (요약 데모)")

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["protan", "deutan", "tritan"],
    format_func=lambda x: {"protan": "Protanopia", "deutan": "Deuteranopia", "tritan": "Tritanopia"}[x],
)

max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)
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
        uploaded_img = Image.open(img_file).convert("RGB")

with col_u2:
    st.subheader("② 사용 방법")
    st.markdown(
        "- 좌측에서 이미지를 업로드하세요.\n"
        "- 색각 유형과 해상도를 사이드바에서 조정하세요.\n"
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

# 3) 보정 적용 -> ndarray(BGR)
corrected = correct_image(cv_small, ctype=ctype)

# 4) 마스크 비적용: 원본/보정 전체 사용
masked_src = cv_small
masked_dst = corrected

# ===== 출력 =====
# (1) 원본/보정 결과
c1, c2 = st.columns([1, 1], gap="medium")
with c1:
    st.subheader("원본")
    st.image(cv_to_pil(masked_src), use_container_width=True)
with c2:
    st.subheader("보정 결과")
    st.image(cv_to_pil(masked_dst), use_container_width=True)

# (2) 전/후 비교(동일 간격)
st.subheader("전/후 비교 (가로 병치)")
c3, c4 = st.columns([1, 1], gap="medium")
with c3:
    st.image(cv_to_pil(masked_src), use_container_width=True, caption="원본")
with c4:
    st.image(cv_to_pil(masked_dst), use_container_width=True, caption=f"보정 ({ctype})")

st.caption("Tip: 사이드바에서 처리 해상도를 조절해 성능/품질을 맞춰보세요.")
