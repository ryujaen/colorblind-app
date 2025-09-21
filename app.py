# app.py — TrueColor (confusion-line 기반 최종본)
from io import BytesIO
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps
from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# ===== Linear <-> sRGB 변환 =====
def _srgb_to_linear(x):
    x = x.astype(np.float32) / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)

def _linear_to_srgb(y):
    a = 0.055
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, y*12.92, (1+a)*(y**(1/2.4)) - a)

# ===== Machado confusion-line 투영 행렬 =====
_M_PROJ = {
    "protan": np.array([[0.0, 2.02344, -2.52581],
                        [0.0, 1.0,      0.0    ],
                        [0.0, 0.0,      1.0    ]], dtype=np.float32),
    "deutan": np.array([[1.0,      0.0,     0.0    ],
                        [0.494207, 0.0,     1.24827],
                        [0.0,      0.0,     1.0    ]], dtype=np.float32),
    "tritan": np.array([[1.0,      0.0,     0.0    ],
                        [0.0,      1.0,     0.0    ],
                        [-0.395913,0.801109, 0.0   ]], dtype=np.float32),
}

def _simulate_confusion(rgb_lin, kind, severity=1.0):
    """confusion-line 시뮬레이션"""
    M = np.eye(3, dtype=np.float32)*(1.0-severity) + _M_PROJ[kind]*severity
    h, w, _ = rgb_lin.shape
    sim = rgb_lin.reshape(-1, 3) @ M.T
    return np.clip(sim.reshape(h, w, 3), 0.0, 1.0)

def daltonize_confusion_line_bgr(img_bgr, kind, alpha=1.0, severity=1.0):
    """confusion-line 기반 보정"""
    # BGR → RGB linear
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_lin = _srgb_to_linear(rgb)

    # confusion-line 시뮬
    sim_lin = _simulate_confusion(rgb_lin, kind, severity)

    # 보정 = 원본 + α*(원본 - 시뮬레이션)
    corrected_lin = rgb_lin + alpha * (rgb_lin - sim_lin)
    corrected_lin = np.clip(corrected_lin, 0.0, 1.0)

    # linear → sRGB → uint8 BGR
    out_rgb = (_linear_to_srgb(corrected_lin) * 255.0).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

# ===== Streamlit 기본 설정 =====
st.set_page_config(page_title="TrueColor", layout="wide")
st.sidebar.title("TrueColor")
st.sidebar.caption("색각 이상자를 위한 색상 보정 웹앱 (confusion-line)")

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["protan", "deutan", "tritan"],
    format_func=lambda x: {"protan":"Protanopia","deutan":"Deuteranopia","tritan":"Tritanopia"}[x]
)
max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)
alpha = st.sidebar.slider("보정 강도 (α)", 0.0, 2.0, 1.0, step=0.1)

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
    st.markdown("- 이미지를 업로드하세요.\n- 색각 유형, 해상도, 보정 강도를 조정하세요.\n- 원본/보정 결과를 비교해보세요.")

st.divider()
if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드해 주세요.")
    st.stop()

# ===== 처리 파이프라인 =====
pil_small = safe_resize(uploaded_img, target_long=max_width)
cv_small = pil_to_cv(pil_small)

# 보정 실행
corrected = daltonize_confusion_line_bgr(cv_small, kind=ctype, alpha=alpha, severity=1.0)

# 보정 차이 (한 번만 표시)
diff = float(np.mean(np.abs(corrected.astype(np.int16) - cv_small.astype(np.int16))))
st.sidebar.write("보정 차이:", round(diff, 3))

# ===== 출력 =====
src_pil, dst_pil = cv_to_pil(cv_small), cv_to_pil(corrected)
c1, c2 = st.columns([1,1], gap="medium")
with c1: st.subheader("원본"); st.image(src_pil, use_column_width=True)
with c2: st.subheader("보정 결과"); st.image(dst_pil, use_column_width=True)

# 전/후 비교
st.subheader("전/후 비교 (가로 병치)")
compare_cv = side_by_side(cv_small, corrected, gap=16)
st.image(cv_to_pil(compare_cv), use_column_width=True)
