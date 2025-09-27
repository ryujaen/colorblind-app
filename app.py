# app.py — TrueColor (Daltonization + Inverse Compensation Adaptive)
import streamlit as st
st.set_page_config(page_title="TrueColor", layout="wide")

from io import BytesIO
import numpy as np
import cv2
from PIL import Image, ImageOps

from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

# =========================
# 0) sRGB <-> linear
# =========================
def _srgb_to_linear(x_uint8: np.ndarray) -> np.ndarray:
    x = x_uint8.astype(np.float32) / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

def _linear_to_srgb(y: np.ndarray) -> np.ndarray:
    a = 0.055
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, y*12.92, (1 + a)*(y**(1/2.4)) - a)

def _apply_matrix(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    return (img.reshape(-1, c) @ M.T).reshape(h, w, c)

# =========================
# 1) 색공간 변환 행렬 (Machado/Vischeck 계열)
# =========================
_RGB2LMS = np.array([
    [17.8824,   43.5161,   4.11935 ],
    [ 3.45565,  27.1554,   3.86714 ],
    [ 0.0299566,0.184309,  1.46709 ]
], dtype=np.float32)

_LMS2RGB = np.array([
    [ 0.0809444479,  -0.130504409,   0.116721066 ],
    [-0.0102485335,   0.0540193266, -0.113614708 ],
    [-0.0003652969,  -0.0041216147,  0.693511405 ]
], dtype=np.float32)

# Ruderman lαβ(대응반대색) 변환 (상대적/정규화판)
# 참고: l은 밝기, α는 적-녹, β는 청-황 축에 대응
_LMS2LAB = np.array([
    [  1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)],
    [  1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
    [  1/np.sqrt(2), -1/np.sqrt(2),  0]
], dtype=np.float32)
# LAB은 로그 LMS 기반이 일반적이지만, 여기선 경량 근사로 사용

# 혼동선 투영(완전 결핍) 행렬을 LMS에서 정의
_M_SIM_LMS = {
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

def _mat_lerp(I: np.ndarray, M: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * I + t * M

# =========================
# 2) 시뮬레이션
# =========================
def simulate_cvd_bgr(img_bgr: np.ndarray, kind: str, severity: float = 1.0) -> np.ndarray:
    kind = "protan" if kind.startswith("prot") else "deutan" if kind.startswith("deut") else "tritan"
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lin = _srgb_to_linear(rgb)
    lms = _apply_matrix(lin, _RGB2LMS)
    M = _mat_lerp(np.eye(3, np.float32), _M_SIM_LMS[kind], severity).astype(np.float32)
    lms_sim = _apply_matrix(lms, M)
    lin_sim = _apply_matrix(lms_sim, _LMS2RGB)
    lin_sim = np.clip(lin_sim, 0.0, 1.0)
    rgb_sim = (_linear_to_srgb(lin_sim) * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_sim, cv2.COLOR_RGB2BGR)

# =========================
# 3) Daltonization (일반 시야에서 구분력 ↑)
# =========================
_C_COMP = {
    "protan": np.array([[0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.7],
                        [0.0, 0.7, 1.0]], dtype=np.float32),
    "deutan": np.array([[1.0, 0.0, 0.7],
                        [0.0, 0.0, 0.0],
                        [0.7, 0.0, 1.0]], dtype=np.float32),
    "tritan": np.array([[1.0, 0.7, 0.0],
                        [0.7, 1.0, 0.0],
                        [0.0, 0.0, 0.0]], dtype=np.float32),
}
def daltonize_bgr(img_bgr: np.ndarray, kind: str, alpha: float, severity: float):
    kind = "protan" if kind.startswith("prot") else "deutan" if kind.startswith("deut") else "tritan"
    sim_bgr = simulate_cvd_bgr(img_bgr, kind, severity)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_sim = cv2.cvtColor(sim_bgr, cv2.COLOR_BGR2RGB)
    lin = _srgb_to_linear(rgb)
    lin_sim = _srgb_to_linear(rgb_sim)
    err = lin - lin_sim
    C = _C_COMP[kind]
    corr = _apply_matrix(err, C)
    out_lin = np.clip(lin + float(alpha) * corr, 0.0, 1.0)
    out_rgb = (_linear_to_srgb(out_lin) * 255.0).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

# =========================
# 4) Inverse Compensation (적응형, 픽셀별)
#     목표: S(corrected) ≈ original  (색각 이상자 시야에서 원본과 비슷)
#     방법: A = L2RGB @ M @ RGB2LMS 의 선형모델을 역문제로 풀되,
#           opponent 축 크기에 따라 픽셀별 λ를 조정해 과보정/단일필터화 방지
# =========================
def _build_A(kind: str, severity: float) -> np.ndarray:
    kind = "protan" if kind.startswith("prot") else "deutan" if kind.startswith("deut") else "tritan"
    M = _mat_lerp(np.eye(3, np.float32), _M_SIM_LMS[kind], severity).astype(np.float32)
    # 전체 파이프라인을 linear-RGB 도메인의 단일 행렬 A로 근사
    A = _LMS2RGB @ M @ _RGB2LMS
    return A.astype(np.float32)

def _opponent_strength_from_rgb_linear(lin: np.ndarray, kind: str) -> np.ndarray:
    """픽셀별 상대 가중치 계산: lαβ의 α(적-녹) 또는 β(청-황) 크기를 사용."""
    lms = _apply_matrix(lin, _RGB2LMS)
    # 로그 근사 안정화 (감마와 범위 차이를 줄이기 위해 소량 오프셋)
    lms_log = np.log1p(1000.0 * np.clip(lms, 0.0, 1.0))  # 안정적 스케일
    lab = _apply_matrix(lms_log, _LMS2LAB)
    kind = "protan" if kind.startswith("prot") else "deutan" if kind.startswith("deut") else "tritan"
    if kind in ("protan", "deutan"):
        strength = np.abs(lab[:, :, 2])  # α축 (여기선 row order [l, β, α]가 될 수 있어 이름 주의)
    else:
        strength = np.abs(lab[:, :, 1])  # β축
    # 0~1로 정규화
    s = strength
    s = s / (np.percentile(s, 99.0) + 1e-6)
    return np.clip(s, 0.0, 1.0)

def inverse_compensate_bgr_adaptive(img_bgr: np.ndarray, kind: str, alpha: float, severity: float) -> np.ndarray:
    """
    적응형 역보정:
      x = argmin ||A x - o||^2 + λ(w) ||x - o||^2
      해: x = (A^T A + λI)^-1 (A^T o + λ o)
      단, λ는 픽셀별로 opponent 강도에 따라 조정 (강한 구분 필요 영역=작은 λ → 더 강한 역보정)
      마지막에 α로 원본과 블렌딩
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lin = _srgb_to_linear(rgb)           # [0..1]
    A = _build_A(kind, severity)         # 3x3
    At = A.T
    AtA = At @ A                         # 3x3 (공통)

    # 픽셀별 λ 계산
    opp = _opponent_strength_from_rgb_linear(lin, kind)  # 0..1
    # λ는 0.001~0.08 범위에서 가변. opp가 클수록(구분 중요) λ ↓ (역보정 강하게)
    lam_min, lam_max = 1e-3, 8e-2
    lam_map = lam_max - opp * (lam_max - lam_min)  # (H,W)

    h, w, _ = lin.shape
    o = lin.reshape(-1, 3)
    # per-pixel 해 구하기
    out = np.empty_like(o)
    I3 = np.eye(3, dtype=np.float32)
    # 벡터화: 각 픽셀마다 3x3 역행렬은 수천 번도 충분히 감당됨(720p 기준)
    lam_flat = lam_map.reshape(-1)
    AtO = (At @ o.T).T  # (N,3)
    for i in range(o.shape[0]):
        lam = lam_flat[i]
        M_inv = np.linalg.inv(AtA + lam * I3).astype(np.float32)
        out[i] = (M_inv @ (AtO[i] + lam * o[i])).astype(np.float32)

    x_lin = out.reshape(h, w, 3)
    x_lin = np.clip(x_lin, 0.0, 1.0)

    # α 블렌딩 (자연스러움)
    a = float(np.clip(alpha, 0.0, 1.0))
    mix_lin = np.clip((1.0 - a) * lin + a * x_lin, 0.0, 1.0)

    out_rgb = (_linear_to_srgb(mix_lin) * 255.0).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

# =========================
# 5) UI
# =========================
st.sidebar.title("TrueColor")
mode = st.sidebar.selectbox(
    "모드 선택",
    options=["Daltonization (일반 시야 구분력↑)", "Inverse compensation (색각 시야 원본≈)"]
)

ctype = st.sidebar.selectbox(
    "색각 유형 선택",
    options=["protan", "deutan", "tritan"],
    format_func=lambda x: {"protan":"Protanopia(적색맹)","deutan":"Deuteranopia(녹색맹)","tritan":"Tritanopia(청색맹)"}[x],
)

alpha = st.sidebar.slider("보정 강도 (α)", 0.0, 1.0, 0.8, step=0.05)
severity = st.sidebar.slider("결함 강도 (severity)", 0.0, 1.0, 1.0, 0.05)
max_width = st.sidebar.slider("처리 해상도 (긴 변 기준 px)", 480, 1280, 720, step=40)
st.sidebar.caption("- Daltonization: 일반 시야에서 색 구분을 강화\n- Inverse: 색각 시야에서 원본과 비슷하게 보이도록 역보정")

st.title("TrueColor — 색각 보정 도구")

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
        "- **모드**를 선택하고 α/결함강도를 조절하세요.\n"
        "- Inverse 모드는 픽셀별로 가변 λ를 사용하여 **색마다 다른 보정**이 적용됩니다."
    )

st.divider()
if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드해 주세요.")
    st.stop()

# =========================
# 6) 처리
# =========================
pil_small = safe_resize(uploaded_img, target_long=max_width)
cv_src = pil_to_cv(pil_small)

if mode.startswith("Daltonization"):
    cv_dst = daltonize_bgr(cv_src, ctype, alpha, severity)
else:
    cv_dst = inverse_compensate_bgr_adaptive(cv_src, ctype, alpha, severity)

# 품질 지표: 선택한 유형의 시야에서 보정 전/후 차이
sim_before = simulate_cvd_bgr(cv_src, ctype, severity)
sim_after  = simulate_cvd_bgr(cv_dst, ctype, severity)
orig_lin = _srgb_to_linear(cv2.cvtColor(sim_before, cv2.COLOR_BGR2RGB))
corr_lin = _srgb_to_linear(cv2.cvtColor(sim_after,  cv2.COLOR_BGR2RGB))
err = np.mean(np.abs(corr_lin - orig_lin))
st.sidebar.write("시야 오차 (↓좋음):", round(float(err), 4))

# =========================
# 7) 출력 + 다운로드
# =========================
c1, c2 = st.columns([1,1], gap="medium")
src_pil = cv_to_pil(cv_src)
dst_pil = cv_to_pil(cv_dst)

with c1:
    st.subheader("원본")
    st.image(src_pil, use_column_width=True)
    buf_src = BytesIO(); src_pil.save(buf_src, format="PNG")
    st.download_button("🖼️ 원본 이미지 다운로드", buf_src.getvalue(),
                       file_name=f"truecolor_original_{max_width}px.png",
                       mime="image/png", use_container_width=True)

with c2:
    st.subheader("보정 결과")
    st.image(dst_pil, use_column_width=True)
    buf_dst = BytesIO(); dst_pil.save(buf_dst, format="PNG")
    st.download_button("✅ 보정 이미지 다운로드", buf_dst.getvalue(),
                       file_name=f"truecolor_{ctype}_{'inv' if mode.startswith('Inverse') else 'dal'}_a{alpha}_s{severity}_{max_width}px.png",
                       mime="image/png", use_column_width=True)

st.subheader("참고: 색각 시야 시뮬레이션 (보정 전/후)")
s1, s2 = st.columns(2)
with s1:
    st.caption("보정 전 — 해당 유형 시야")
    st.image(cv_to_pil(sim_before), use_column_width=True)
with s2:
    st.caption("보정 후 — 해당 유형 시야")
    st.image(cv_to_pil(sim_after), use_column_width=True)

st.subheader("전/후 비교 (가로 병치)")
compare_cv = side_by_side(cv_src, cv_dst, gap=16)
compare_pil = cv_to_pil(compare_cv)
st.image(compare_pil, use_column_width=True)
buf_cmp = BytesIO(); compare_pil.save(buf_cmp, format="PNG")
st.download_button("↔️ 전/후 비교(병치) 이미지 다운로드", buf_cmp.getvalue(),
                   file_name=f"truecolor_compare_{ctype}_{'inv' if mode.startswith('Inverse') else 'dal'}_{max_width}px.png",
                   mime="image/png", use_column_width=True)

st.caption("Tip) Inverse 모드에서 α를 낮추면 자연스러움↑, 올리면 색각 시야 일치도↑. 단색보다 다양한 색이 있는 이미지에서 효과가 분명합니다.")
