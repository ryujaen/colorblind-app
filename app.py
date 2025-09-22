# app.py — TrueColor (inverse-simulation compensation + downloads)
import streamlit as st
st.set_page_config(page_title="TrueColor", layout="wide")

from io import BytesIO
import numpy as np
import cv2
from PIL import Image, ImageOps

from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# =========================
# 1) 보정 핵심 유틸 (inverse simulation)
# =========================

# Machado 2009 confusion-line projection matrices (linear RGB domain)
_M_PROJ = {
    "protan": np.array([[0.0,      2.02344, -2.52581],
                        [0.0,      1.0,      0.0    ],
                        [0.0,      0.0,      1.0    ]], dtype=np.float32),
    "deutan": np.array([[1.0,      0.0,      0.0    ],
                        [0.494207, 0.0,      1.24827],
                        [0.0,      0.0,      1.0    ]], dtype=np.float32),
    "tritan": np.array([[1.0,      0.0,      0.0    ],
                        [0.0,      1.0,      0.0    ],
                        [-0.395913,0.801109, 0.0    ]], dtype=np.float32),
}

def _srgb_to_linear(x_uint8: np.ndarray) -> np.ndarray:
    x = x_uint8.astype(np.float32) / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)

def _linear_to_srgb(y: np.ndarray) -> np.ndarray:
    a = 0.055
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, y*12.92, (1+a)*(y**(1/2.4)) - a)

def _confusion_matrix(kind: str, severity: float) -> np.ndarray:
    """S(x) = M(severity) @ x  (linear RGB)"""
    k = kind.lower()
    base = _M_PROJ["protan"] if k.startswith("prot") else \
           _M_PROJ["deutan"] if k.startswith("deut") else \
           _M_PROJ["tritan"]
    I = np.eye(3, dtype=np.float32)
    return (1.0 - float(severity)) * I + float(severity) * base

def compensate_confusion_inverse_bgr(img_bgr: np.ndarray,
                                     kind: str,
                                     alpha: float = 1.0,
                                     severity: float = 1.0) -> np.ndarray:
    """
    S(corrected) ≈ original 을 목표로 하는 안정화된 역보정.
    - 정규화 의사역행렬:  (M^T M + λI)^(-1) M^T  (λ는 severity에 따라 가변)
    - alpha는 0..1 범위에서 선형 보간(lerp)
    """
    # BGR → RGB → linear
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lin = _srgb_to_linear(rgb)

    # confusion matrix & regularized inverse
    M = _confusion_matrix(kind, float(severity)).astype(np.float32)
    I = np.eye(3, dtype=np.float32)
    lam = 1e-3 + 5e-2 * float(severity)        # 예: 0.001 ~ 0.051
    Minv_reg = np.linalg.inv(M.T @ M + lam * I) @ M.T

    h, w, _ = lin.shape
    corr_lin = lin.reshape(-1, 3) @ Minv_reg.T
    corr_lin = corr_lin.reshape(h, w, 3)
    corr_lin = np.clip(corr_lin, 0.0, 1.0)

    # alpha 0..1 lerp
    a = float(np.clip(alpha, 0.0, 1.0))
    out_lin = (1.0 - a) * lin + a * corr_lin
    out_lin = np.clip(out_lin, 0.0, 1.0)

    out_rgb = (_linear_to_srgb(out_lin) * 255.0).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

# =========================
# 2) CSS (selectbox 커서 고정)
# =========================
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

# =========================
# 3) 사이드바
# =========================
st.sidebar.title("TrueColor")
st.sidebar.caption("색각 이상자를 위한 색상 보정 웹앱 (inverse simulation)")

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["protan", "deutan", "tritan"],
    format_func=lambda x: {"protan":"Protanopia","deutan":"Deuteranopia","tritan":"Tritanopia"}[x],
)

alpha = st.sidebar.slider(
    "보정 강도 (α)", 0.0, 1.0, 0.8, step=0.05,
    help="0.0은 원본 유지, 1.0은 역보정 100% 적용"
)
severity = st.sidebar.slider("결함 강도 (severity)", 0.0, 1.0, 1.0, 0.05,
                             help="1.0은 완전 색각결함 가정, 0.5는 약한 결함")
max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)
st.sidebar.divider()

# =========================
# 4) 본문
# =========================
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
    st.markdown(
        "- 이미지를 업로드하고 사이드바에서 유형/강도/결함강도를 조절하세요.\n"
        "- 보정 결과는 **색각이상자 시야에서 원본과 같아지도록** 계산됩니다."
    )

st.divider()
if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드해 주세요.")
    st.stop()

# =========================
# 5) 처리 파이프라인
# =========================
pil_small = safe_resize(uploaded_img, target_long=max_width)
cv_src = pil_to_cv(pil_small)

# 핵심: inverse-simulation compensation
cv_dst = compensate_confusion_inverse_bgr(cv_src, kind=ctype, alpha=alpha, severity=severity)

# 보정 품질 지표(한 번만): 색각이상자 시야에서의 오차 || S(corrected) - original ||
# (linear RGB에서 평가)
M_eval = _confusion_matrix(ctype, severity)
orig_lin = _srgb_to_linear(cv2.cvtColor(cv_src, cv2.COLOR_BGR2RGB))
corr_lin = _srgb_to_linear(cv2.cvtColor(cv_dst, cv2.COLOR_BGR2RGB))
err = np.mean(np.abs(corr_lin.reshape(-1,3) @ M_eval.T - orig_lin.reshape(-1,3)))
st.sidebar.write("보정 차이(시야 오차):", round(float(err), 4))

# =========================
# 6) 출력 + 다운로드
# =========================
c1, c2 = st.columns([1,1], gap="medium")
src_pil = cv_to_pil(cv_src)
dst_pil = cv_to_pil(cv_dst)

with c1:
    st.subheader("원본")
    st.image(src_pil, use_column_width=True)
    # 다운로드(원본)
    buf_src = BytesIO()
    src_pil.save(buf_src, format="PNG")
    st.download_button(
        "🖼️ 원본 이미지 다운로드",
        data=buf_src.getvalue(),
        file_name=f"truecolor_original_{max_width}px.png",
        mime="image/png",
        use_container_width=True,
    )

with c2:
    st.subheader("보정 결과")
    st.image(dst_pil, use_column_width=True)
    # 다운로드(보정)
    buf_dst = BytesIO()
    dst_pil.save(buf_dst, format="PNG")
    st.download_button(
        "✅ 보정 이미지 다운로드",
        data=buf_dst.getvalue(),
        file_name=f"truecolor_{ctype}_alpha{alpha}_sev{severity}_{max_width}px.png",
        mime="image/png",
        use_container_width=True,
    )

# 전/후 비교 (가로 병치) + 다운로드
st.subheader("전/후 비교 (가로 병치)")
compare_cv = side_by_side(cv_src, cv_dst, gap=16)
compare_pil = cv_to_pil(compare_cv)
st.image(compare_pil, use_column_width=True)

comp_buf = BytesIO()
compare_pil.save(comp_buf, format="PNG")
st.download_button(
    "↔️ 전/후 비교(병치) 이미지 다운로드",
    data=comp_buf.getvalue(),
    file_name=f"truecolor_compare_{ctype}_alpha{alpha}_sev{severity}_{max_width}px.png",
    mime="image/png",
    use_container_width=True,
)

st.caption("Tip: α(보정 강도)와 severity(결함 강도)를 조절해 자연스러움과 일치도를 맞춰보세요.")
