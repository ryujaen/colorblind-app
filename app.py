# app.py — TrueColor (Daltonization final)
import streamlit as st
st.set_page_config(page_title="TrueColor", layout="wide")

from io import BytesIO
import numpy as np
import cv2
from PIL import Image, ImageOps

from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# =========================
# 1) 색 공간/행렬 유틸
# =========================

# sRGB <-> linear
def _srgb_to_linear(x_uint8: np.ndarray) -> np.ndarray:
    x = x_uint8.astype(np.float32) / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

def _linear_to_srgb(y: np.ndarray) -> np.ndarray:
    a = 0.055
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, y*12.92, (1 + a)*(y**(1/2.4)) - a)

# Machado/Vischeck 계열에서 널리 쓰는 RGB↔LMS 변환 행렬
_RGB2LMS = np.array([
    [17.8824,  43.5161,   4.11935],
    [ 3.45565, 27.1554,   3.86714],
    [ 0.0299566, 0.184309, 1.46709]
], dtype=np.float32)

_LMS2RGB = np.array([
    [ 0.0809444479,  -0.130504409,   0.116721066],
    [-0.0102485335,   0.0540193266, -0.113614708],
    [-0.000365296938,-0.00412161469, 0.693511405]
], dtype=np.float32)

# 색각결함(완전형) 시뮬레이션 행렬 (LMS 도메인)
# severity(0..1)로 I~M을 보간해서 부분결함도 표현
_M_SIM_LMS = {
    "protan": np.array([
        [0.0,      2.02344, -2.52581],
        [0.0,      1.0,      0.0    ],
        [0.0,      0.0,      1.0    ]
    ], dtype=np.float32),
    "deutan": np.array([
        [1.0,      0.0,      0.0    ],
        [0.494207, 0.0,      1.24827],
        [0.0,      0.0,      1.0    ]
    ], dtype=np.float32),
    "tritan": np.array([
        [1.0,      0.0,      0.0    ],
        [0.0,      1.0,      0.0    ],
        [-0.395913,0.801109, 0.0    ]
    ], dtype=np.float32),
}

def _mat_lerp(I: np.ndarray, M: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * I + t * M

def _apply_matrix(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    out = img.reshape(-1, c) @ M.T
    return out.reshape(h, w, c)

# =========================
# 2) 시뮬레이션 + Daltonization 보정
# =========================

def simulate_cvd_bgr(img_bgr: np.ndarray, kind: str, severity: float = 1.0) -> np.ndarray:
    """BGR(uint8) → 시뮬레이션 BGR(uint8), LMS에서 confusion-line 투영."""
    # BGR→RGB→linear→LMS
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lin = _srgb_to_linear(rgb)
    lms = _apply_matrix(lin, _RGB2LMS)

    I = np.eye(3, dtype=np.float32)
    M_full = _M_SIM_LMS["protan" if kind.startswith("prot") else
                        "deutan" if kind.startswith("deut") else
                        "tritan"]
    M = _mat_lerp(I, M_full, severity).astype(np.float32)

    lms_sim = _apply_matrix(lms, M)
    lin_sim = _apply_matrix(lms_sim, _LMS2RGB)
    lin_sim = np.clip(lin_sim, 0.0, 1.0)
    rgb_sim = (_linear_to_srgb(lin_sim) * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_sim, cv2.COLOR_RGB2BGR)

# 채널 보정 주입(표준 휴리스틱): 결핍 채널의 구분력을 다른 채널에서 끌어와 보강
# 문헌/레퍼런스 구현에서 자주 쓰이는 간단 행렬
_C_COMP = {
    # protan: R 결핍 → G/B 차이를 R/다른 채널에 주입
    "protan": np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.7],
        [0.0, 0.7, 1.0]
    ], dtype=np.float32),
    # deutan: G 결핍
    "deutan": np.array([
        [1.0, 0.0, 0.7],
        [0.0, 0.0, 0.0],
        [0.7, 0.0, 1.0]
    ], dtype=np.float32),
    # tritan: B 결핍
    "tritan": np.array([
        [1.0, 0.7, 0.0],
        [0.7, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32),
}

def daltonize_bgr(img_bgr: np.ndarray, kind: str, alpha: float = 0.8, severity: float = 1.0) -> np.ndarray:
    """
    Daltonization 파이프라인:
      1) CVD 시뮬레이션
      2) 오차 e = original - simulated
      3) 보정 m = C(kind) @ e  (채널 간 주입)
      4) 보정 결과 = clip(original + alpha * m)
    """
    kind = kind.lower()
    kind = "protan" if kind.startswith("prot") else "deutan" if kind.startswith("deut") else "tritan"

    # 1) simulate
    sim_bgr = simulate_cvd_bgr(img_bgr, kind, severity)

    # 2) error in linear RGB로 계산하면 과보정 줄어듦
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_sim = cv2.cvtColor(sim_bgr, cv2.COLOR_BGR2RGB)
    lin = _srgb_to_linear(rgb)
    lin_sim = _srgb_to_linear(rgb_sim)
    err = lin - lin_sim  # [-1..1] 주변

    # 3) channel compensation
    C = _C_COMP[kind]
    corr = _apply_matrix(err, C)

    # 4) blend
    a = float(np.clip(alpha, 0.0, 1.0))
    lin_corr = np.clip(lin + a * corr, 0.0, 1.0)

    out_rgb = (_linear_to_srgb(lin_corr) * 255.0).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR), sim_bgr

# =========================
# 3) CSS (selectbox 커서 고정)
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
# 4) 사이드바
# =========================
st.sidebar.title("TrueColor")
st.sidebar.caption("색각 이상자를 위한 색상 보정 웹앱 (Daltonization)")

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["protan", "deutan", "tritan"],
    format_func=lambda x: {"protan":"Protanopia(적색맹)","deutan":"Deuteranopia(녹색맹)","tritan":"Tritanopia(청색맹)"}[x],
)

alpha = st.sidebar.slider("보정 강도 (α)", 0.0, 1.0, 0.8, step=0.05,
                          help="0.0=보정 끔, 1.0=보정 100%")
severity = st.sidebar.slider("결함 강도 (severity)", 0.0, 1.0, 1.0, 0.05,
                             help="시뮬레이션/오차계산에 쓰이는 결함 강도")
max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)
st.sidebar.divider()

# =========================
# 5) 본문
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
        "- 이미지를 업로드하고 좌측에서 **유형/α/결함강도**를 조절하세요.\n"
        "- 보정은 Daltonization으로, **색각 이상 시야에서의 구분력**을 높이는 데 초점을 둡니다."
    )

st.divider()
if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드해 주세요.")
    st.stop()

# =========================
# 6) 처리 파이프라인
# =========================
pil_small = safe_resize(uploaded_img, target_long=max_width)
cv_src = pil_to_cv(pil_small)

cv_dst, cv_sim = daltonize_bgr(cv_src, kind=ctype, alpha=alpha, severity=severity)

# 품질 지표(참고): 시뮬레이션 공간에서의 평균 절대오차
sim_again = simulate_cvd_bgr(cv_dst, ctype, severity)
orig_sim = simulate_cvd_bgr(cv_src, ctype, severity)
orig_lin = _srgb_to_linear(cv2.cvtColor(orig_sim, cv2.COLOR_BGR2RGB))
corr_lin = _srgb_to_linear(cv2.cvtColor(sim_again, cv2.COLOR_BGR2RGB))
err = np.mean(np.abs(corr_lin - orig_lin))
st.sidebar.write("시야 오차(↓좋음):", round(float(err), 4))

# =========================
# 7) 출력 + 다운로드
# =========================
c1, c2 = st.columns([1,1], gap="medium")
src_pil = cv_to_pil(cv_src)
dst_pil = cv_to_pil(cv_dst)

with c1:
    st.subheader("원본")
    st.image(src_pil, use_column_width=True)
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
    buf_dst = BytesIO()
    dst_pil.save(buf_dst, format="PNG")
    st.download_button(
        "✅ 보정 이미지 다운로드",
        data=buf_dst.getvalue(),
        file_name=f"truecolor_{ctype}_alpha{alpha}_sev{severity}_{max_width}px.png",
        mime="image/png",
        use_container_width=True,
    )

st.subheader("참고: 색각 시뮬레이션(보정 전/후)")
s1, s2 = st.columns(2)
with s1:
    st.caption("보정 전 — 해당 유형 시야 시뮬레이션")
    st.image(cv_to_pil(cv_sim), use_column_width=True)
with s2:
    st.caption("보정 후 — 해당 유형 시야 시뮬레이션")
    st.image(cv_to_pil(sim_again), use_column_width=True)

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

st.caption("Tip: α는 보정 강도, severity는 시야 시뮬레이션 강도입니다. 자연스러움과 구분력 사이에서 균형을 찾아보세요.")
