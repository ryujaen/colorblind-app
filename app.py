import streamlit as st
st.set_page_config(page_title="TrueColor", layout="wide")

import numpy as np
import cv2
from PIL import Image, ImageOps

from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# ===== confusion-line 기반 보정 함수 =====
def daltonize_confusion_line_bgr(img_bgr, kind, alpha=1.0, severity=1.0):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if kind == "protan":
        # R 결핍 → G, B로 보정
        r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
        g2 = g + alpha * 0.7 * (r - g)
        b2 = b + alpha * 0.7 * (r - b)
        out = np.stack([r, g2, b2], axis=-1)
    elif kind == "deutan":
        # G 결핍 → R, B로 보정
        r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
        r2 = r + alpha * 0.7 * (g - r)
        b2 = b + alpha * 0.7 * (g - b)
        out = np.stack([r2, g, b2], axis=-1)
    else:  # tritan
        # B 결핍 → R, G로 보정
        r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
        r2 = r + alpha * 0.7 * (b - r)
        g2 = g + alpha * 0.7 * (b - g)
        out = np.stack([r2, g2, b], axis=-1)

    out = np.clip(out, 0.0, 1.0)
    out_rgb = (out * 255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

# ===== CSS (selectbox 커서 고정) =====
st.markdown(
    """
    <style>
    div[data-baseweb="select"] * { cursor: default !important; user-select: none !important; }
    div[data-baseweb="select"] input { caret-color: transparent !important; }
    div[data-baseweb="select"] [role="combobox"] { cursor: default !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== 사이드바 =====
st.sidebar.title("TrueColor")
st.sidebar.caption("색각 이상자를 위한 색상 보정 웹앱 (confusion-line)")

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["protan", "deutan", "tritan"],
    format_func=lambda x: {"protan":"Protanopia","deutan":"Deuteranopia","tritan":"Tritanopia"}[x],
)

alpha = st.sidebar.slider("보정 강도 (α)", 0.0, 2.0, 1.0, step=0.1)
max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)
st.sidebar.divider()

# ===== 본문 =====
st.title("TrueColor – 색상 보정 전/후 비교")

col_u1, col_u2 = st.columns(2)
uploaded_img = None
with col_u1:
    st.subheader("① 이미지/사진 입력")
    img_file = st.file_uploader("이미지 업로드 (JPG/PNG)", type=["jpg","jpeg","png"])
    if img_file:
        uploaded_img = ImageOps.exif_transpose(Image.open(img_file)).convert("RGB")

with col_u2:
    st.subheader("② 사용 방법")
    st.markdown("- 이미지를 업로드하고 사이드바에서 유형/강도를 조절하세요.\n- 아래에서 원본/보정 결과를 비교할 수 있습니다.")

st.divider()

if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드해 주세요.")
    st.stop()

# ===== 처리 파이프라인 =====
pil_small = safe_resize(uploaded_img, target_long=max_width)
cv_src = pil_to_cv(pil_small)

# confusion-line 기반 보정
cv_dst = daltonize_confusion_line_bgr(cv_src, kind=ctype, alpha=alpha, severity=1.0)

# 보정 차이 (한 번만)
diff = float(np.mean(np.abs(cv_dst.astype(np.int16) - cv_src.astype(np.int16))))
st.sidebar.write("보정 차이:", round(diff, 3))

# ===== 출력 =====
c1, c2 = st.columns([1,1], gap="medium")
with c1:
    st.subheader("원본")
    st.image(cv_to_pil(cv_src), use_column_width=True)
with c2:
    st.subheader("보정 결과")
    st.image(cv_to_pil(cv_dst), use_column_width=True)

st.subheader("전/후 비교 (가로 병치)")
st.image(cv_to_pil(side_by_side(cv_src, cv_dst, gap=16)), use_column_width=True)

st.caption("Tip: 사이드바에서 해상도와 보정 강도를 조절해 효과를 확인하세요.")
