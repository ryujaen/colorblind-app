# app.py — TrueColor (Confusion-line LMS daltonization, clean final)
from io import BytesIO
import numpy as np
import cv2
import streamlit as st
from PIL import ImageOps

from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# ================== Confusion-line (Brettel/Machado style) ==================
# sRGB ↔ Linear
def _srgb_to_lin(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

def _lin_to_srgb(y: np.ndarray) -> np.ndarray:
    a = 0.055
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, y*12.92, (1 + a)*(y**(1/2.4)) - a)

# RGB↔LMS (Hunt–Pointer–Estevez 계열, Machado 구현에서 널리 사용)
_M_RGB2LMS = np.array([
    [0.31399022, 0.63951294, 0.04649755],
    [0.15537241, 0.75789446, 0.08670142],
    [0.01775239, 0.10944209, 0.87256922],
], dtype=np.float32)

_M_LMS2RGB = np.array([
    [ 5.47221206, -4.64196010,  0.16963708],
    [-1.12524190,  2.29317094, -0.16789520],
    [ 0.02980165, -0.19318073,  1.16364789],
], dtype=np.float32)

def _rgb_lin_to_lms(rgb_lin: np.ndarray) -> np.ndarray:
    h, w, _ = rgb_lin.shape
    return (rgb_lin.reshape(-1, 3) @ _M_RGB2LMS.T).reshape(h, w, 3)

def _lms_to_rgb_lin(lms: np.ndarray) -> np.ndarray:
    h, w, _ = lms.shape
    return (lms.reshape(-1, 3) @ _M_LMS2RGB.T).reshape(h, w, 3)

# 완전 결함 시뮬레이션 근사(안정/빠른 버전)
def _simulate_brettel_lms(lms: np.ndarray, kind: str, severity: float = 1.0) -> np.ndarray:
    L, M, S = lms[..., 0], lms[..., 1], lms[..., 2]
    if kind == "protan":          # L 결핍
        Ls = 0.0*L + 1.05118294*M - 0.05116099*S
        Ms = M
        Ss = S
    elif kind == "deutan":        # M 결핍
        Ls = L
        Ms = 0.0*M + 0.95130920*L + 0.04866992*S
        Ss = S
    else:                          # tritan: S 결핍
        Ls = L
        Ms = M
        Ss = 0.0*S + -0.86744736*L + 1.86727089*M
    sim = np.stack([Ls, Ms, Ss], axis=-1)
    return np.clip(lms*(1.0 - severity) + sim*severity, 0.0, None)

def daltonize_confusion_line_bgr(img_bgr: np.ndarray, kind: str,
                                 alpha: float = 1.0, severity: float = 1.0) -> np.ndarray:
    """
    Confusion-line 기반 daltonization.
    입력/출력: BGR uint8
    alpha: 에러 재분배 강도 (0..2 권장)
    severity: 결함 시뮬 강도 (0..1), 1.0=완전 결함 가정
    """
    # BGR→RGB(sRGB)→linear
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_lin = _srgb_to_lin(rgb)

    # linear RGB → LMS
    lms = _rgb_lin_to_lms(rgb_lin)

    # 결함 시뮬레이션 & 에러
    lms_sim = _simulate_brettel_lms(lms, kind, severity=severity)
    err = lms - lms_sim
    L, M, S = lms[..., 0], lms[..., 1], lms[..., 2]
    eL, eM, eS = err[..., 0], err[..., 1], err[..., 2]

    # 에러를 보이는 채널로 재분배(과한 색붕괴 방지 위해 가중 분배)
    if kind == "protan":      # L 결핍 → M,S로
        L2 = L
        M2 = M + alpha * 0.6 * eL
        S2 = S + alpha * 0.4 * eL
    elif kind == "deutan":    # M 결핍 → L,S로
        L2 = L + alpha * 0.6 * eM
        M2 = M
        S2 = S + alpha * 0.4 * eM
    else:                     # tritan: S 결핍 → L,M로
        L2 = L + alpha * 0.5 * eS
        M2 = M + alpha * 0.5 * eS
        S2 = S

    lms_corr = np.stack([L2, M2, S2], axis=-1)
    lms_corr = np.clip(lms_corr, 0.0, None)

    # LMS→linear RGB → sRGB → BGR
    rgb_lin_out = _lms_to_rgb_lin(lms_corr)
    rgb_out = (_lin_to_srgb(rgb_lin_out) * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)

# ================== Streamlit UI ==================
st.set_page_config(page_title="TrueColor", layout="wide")

# selectbox 커서 스타일
st.markdown("""
<style>
div[data-baseweb="select"] { cursor: default !important; }
div[data-baseweb="select"] * { cursor: default !important; }
</style>
""", unsafe_allow_html=True)

# 사이드바
st.sidebar.title("TrueColor")
st.sidebar.caption("색각 이상자를 위한 색상 보정 웹앱 (요약 데모)")

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["protan", "deutan", "tritan"],
    format_func=lambda x: {"protan": "Protanopia",
                           "deutan": "Deuteranopia",
                           "tritan": "Tritanopia"}[x],
)

max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)
alpha = st.sidebar.slider("보정 강도 (α)", 0.0, 2.0, 1.0, step=0.1,
                          help="0.0은 원본 유지, 1.0은 기본 보정, 2.0은 보정을 두 배 적용")

st.sidebar.divider()

# 본문
st.title("TrueColor – 색상 보정 전/후 비교")
st.write("**이미지 업로드 → 보정 적용 → 전/후 비교**")

col_u1, col_u2 = st.columns(2)
uploaded_img = None
with col_u1:
    st.subheader("① 이미지/사진 입력")
    img_file = st.file_uploader("이미지 업로드 (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if img_file:
        uploaded_img = ImageOps.exif_transpose(ImageOps.open(img_file) if hasattr(ImageOps, 'open') else __import__('PIL').Image.open(img_file)).convert("RGB")
with col_u2:
    st.subheader("② 사용 방법")
    st.markdown("- 좌측에서 이미지를 업로드하세요.\n- 색각 유형, 해상도, 보정 강도를 사이드바에서 조정하세요.\n- 아래에서 원본/보정 결과를 비교하고 다운로드할 수 있습니다.")

st.divider()

if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드해 주세요.")
    st.stop()

# 처리 파이프라인
pil_small = safe_resize(uploaded_img, target_long=max_width)   # PIL
cv_small = pil_to_cv(pil_small)                                # BGR uint8

kind = "protan" if ctype.startswith("prot") else ("deutan" if ctype.startswith("deut") else "tritan")
corrected = daltonize_confusion_line_bgr(cv_small, kind=kind, alpha=alpha, severity=1.0)

# 변화량(디버깅)
diff = float(np.mean(np.abs(corrected.astype(np.int16) - cv_small.astype(np.int16))))
st.sidebar.write("보정 차이:", round(diff, 3))

# 출력 & 다운로드
masked_src, masked_dst = cv_small, corrected
src_pil, dst_pil = cv_to_pil(masked_src), cv_to_pil(masked_dst)

c1, c2 = st.columns([1, 1], gap="medium")
with c1:
    st.subheader("원본")
    st.image(src_pil, use_column_width=True)
    buf_src = BytesIO(); src_pil.save(buf_src, format="PNG")
    st.download_button("🖼️ 원본 이미지 다운로드", data=buf_src.getvalue(),
                       file_name=f"truecolor_original_{max_width}px.png", mime="image/png")
with c2:
    st.subheader("보정 결과")
    st.image(dst_pil, use_column_width=True)
    buf_dst = BytesIO(); dst_pil.save(buf_dst, format="PNG")
    st.download_button("✅ 보정 이미지 다운로드", data=buf_dst.getvalue(),
                       file_name=f"truecolor_{kind}_alpha{alpha}_{max_width}px.png", mime="image/png")

st.subheader("전/후 비교 (가로 병치)")
compare_cv = side_by_side(masked_src, masked_dst, gap=16)
compare_pil = cv_to_pil(compare_cv)
comp_buf = BytesIO(); compare_pil.save(comp_buf, format="PNG")
st.image(compare_pil, use_column_width=True)
st.download_button("↔️ 전/후 비교(병치) 다운로드", data=comp_buf.getvalue(),
                   file_name=f"truecolor_compare_{kind}_alpha{alpha}_{max_width}px.png", mime="image/png")

st.caption("Tip: α(보정 강도)를 0.8~1.2 사이에서 조절하면 자연스럽습니다.")
