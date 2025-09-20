# app.py (상단)
import numpy as np
from PIL import Image
import streamlit as st

# 반드시 첫 Streamlit 호출
st.set_page_config(page_title="TrueColor", layout="wide", initial_sidebar_state="expanded")

# 아래부터 다른 import/디버그/출력
from daltonize import correct_image, SUPPORTED_TYPES
from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side
# (디버그가 필요하면 set_page_config 다음 줄 이후에)
# import image_utils as IU
# st.caption(f"image_utils: {getattr(IU,'__file__','?')}")
# st.caption(f"has side_by_side: {'side_by_side' in dir(IU)}")

import image_utils as IU
st.caption(f"side_by_side exists: {'side_by_side' in dir(IU)}")

# image_utils에서 가능한 건 가져오고, 누락된 건 로컬 폴백 정의
try:
    from image_utils import (
        pil_to_cv, cv_to_pil, safe_resize, apply_circle_mask, side_by_side
    )
except ImportError:
    from image_utils import (
        pil_to_cv, cv_to_pil, safe_resize, apply_circle_mask
    )
    # 폴백 side_by_side (BGR ndarray 기준)
    import cv2
    def side_by_side(left, right, gap: int = 16):
        L = left if isinstance(left, np.ndarray) else pil_to_cv(left)
        R = right if isinstance(right, np.ndarray) else pil_to_cv(right)
        h = max(L.shape[0], R.shape[0])
        def _resize_h(a, h):
            scale = h / a.shape[0]
            w = max(1, int(a.shape[1] * scale))
            return cv2.resize(a, (w, h), interpolation=cv2.INTER_LANCZOS4)
        Lr, Rr = _resize_h(L, h), _resize_h(R, h)
        out = np.full((h, Lr.shape[1] + gap + Rr.shape[1], 3), 255, np.uint8)
        out[:, :Lr.shape[1]] = Lr[:, :, :3]
        out[:, Lr.shape[1] + gap:] = Rr[:, :, :3]
        return out
# --- end imports ---

try:
    st.set_page_config(page_title="TrueColor", layout="wide")
except st.errors.StreamlitAPIException:
    pass  # 이미 호출된 경우 무시


# ===== 사이드바 =====
st.sidebar.title("TrueColor")
st.sidebar.caption("색각 이상자를 위한 색상 보정 웹앱 (요약 데모)")

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["protan", "deutan", "tritan"],
    format_func=lambda x: {"protan": "Protanopia", "deutan": "Deuteranopia", "tritan": "Tritanopia"}[x],
)

#mask_bg = st.sidebar.select_slider("원형 마스크 배경 밝기", options=list(range(160, 241, 10)), value=200)
max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)

st.sidebar.divider()
#use_camera = st.sidebar.toggle("브라우저 카메라 사용", value=False, help="브라우저가 지원될 때 권장(st.camera_input).")

# --- debug 확인용 ---
import image_utils as IU
st.caption(f"image_utils path: {getattr(IU,'__file__','?')}")
st.caption(f"has side_by_side: {'side_by_side' in dir(IU)}")
# --- 여기까지 ---

# ===== 본문 =====
st.title("TrueColor – 색상 보정 전/후 비교")
st.write("**업로드(또는 카메라) → 보정 적용 → 전/후 비교**")

col_u1, col_u2 = st.columns(2)
uploaded_img = None

with col_u1:
    st.subheader("① 이미지/사진 입력")
    img_file = st.file_uploader("이미지 업로드 (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if img_file:
        uploaded_img = Image.open(img_file).convert("RGB")

'''
with col_u2:
    st.subheader("② 카메라 입력 (옵션)")
    if use_camera:
        cam_buf = st.camera_input("카메라로 촬영")
        if cam_buf:
            uploaded_img = Image.open(cam_buf).convert("RGB")
'''

st.divider()

if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드하거나, 우측에서 카메라로 촬영해 주세요.")
    st.stop()

# ===== 처리 파이프라인 =====
# 1) 안전 리사이즈(속도 개선) -> PIL
pil_small = safe_resize(uploaded_img, target_long=max_width)

# 2) OpenCV 배열로 변환 -> ndarray(BGR)
cv_small = pil_to_cv(pil_small)

# 3) 보정 적용 -> ndarray(BGR)
corrected = correct_image(cv_small, ctype=ctype)

# 4) 마스크 비적용 (그냥 원본/보정 전체 사용)
masked_src = cv_small
masked_dst = corrected

# 5) 전/후 합성 -> ndarray(BGR)
compare = side_by_side(masked_src, masked_dst)

# ===== 출력 =====
c1, c2 = st.columns([1, 1], gap="medium")
with c1:
    st.subheader("원본 (마스크 적용)")
    st.image(cv_to_pil(masked_src), use_container_width=True)
with c2:
    st.subheader("보정 결과 (마스크 적용)")
    st.image(cv_to_pil(masked_dst), use_container_width=True)

# 전/후 비교 (가로 병치)
st.subheader("전/후 비교 (가로 병치)")
c3, c4 = st.columns([1, 1], gap="medium")
with c3:
    st.image(cv_to_pil(masked_src), use_container_width=True, caption="원본")
with c4:
    st.image(cv_to_pil(masked_dst), use_container_width=True, caption=f"보정 ({ctype})")
