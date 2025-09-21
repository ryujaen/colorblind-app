# app.py — TrueColor (정리/보강 최종본)
from io import BytesIO
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps

from daltonize import correct_image
from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side


# Machado 2009에서 널리 쓰이는 RGB 투영 행렬(선형 RGB에서 동작)
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

def _srgb_to_linear(x):
    x = x.astype(np.float32)
    x = x / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)

def _linear_to_srgb(y):
    a = 0.055
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, y*12.92, (1+a)*(y**(1/2.4)) - a)

def _simulate_confusion_linear(rgb_lin, kind, severity=1.0):
    """
    rgb_lin: HxWx3 (linear RGB, 0..1)
    kind: 'protan'|'deutan'|'tritan'
    severity: 0..1 (1에 가까울수록 완전 결함 시뮬)
    """
    M = np.eye(3, dtype=np.float32)*(1.0 - severity) + _M_PROJ[kind]*(severity)
    h, w, _ = rgb_lin.shape
    sim = rgb_lin.reshape(-1, 3) @ M.T
    sim = sim.reshape(h, w, 3)
    return np.clip(sim, 0.0, 1.0)

def daltonize_confusion_line_bgr(img_bgr, kind, alpha=1.0, severity=1.0):
    """
    1) confusion-line 시뮬레이션
    2) error = original - simulated
    3) error를 '보이는 채널'로 재분배하여 보정 (단순/안정 매핑)
    입력:  BGR uint8, 출력: BGR uint8
    """
    # BGR uint8 -> RGB linear
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_lin = _srgb_to_linear(rgb)

    # 시뮬레이션
    sim_lin = _simulate_confusion_linear(rgb_lin, kind, severity=severity)

    # 에러
    err = rgb_lin - sim_lin

    r, g, b = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    er, eg, eb = err[..., 0], err[..., 1], err[..., 2]

    # 에러 재분배(간단/안정): 결핍 채널의 정보를 다른 두 채널에 가중 유입
    if kind == "protan":
        # R 결핍 → G,B로 보정
        g2 = g + alpha * 0.7 * er
        b2 = b + alpha * 0.7 * er
        r2 = r
    elif kind == "deutan":
        # G 결핍 → R,B로 보정
        r2 = r + alpha * 0.7 * eg
        b2 = b + alpha * 0.7 * eg
        g2 = g
    else:  # 'tritan'
        # B 결핍 → R,G로 보정
        r2 = r + alpha * 0.7 * eb
        g2 = g + alpha * 0.7 * eb
        b2 = b

    out_lin = np.stack([r2, g2, b2], axis=-1)
    out_lin = np.clip(out_lin, 0.0, 1.0)

    # linear -> sRGB -> BGR
    out_rgb = (_linear_to_srgb(out_lin) * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr

# ===== Streamlit 기본 설정 =====
st.set_page_config(page_title="TrueColor", layout="wide")

# ===== CSS (selectbox 위에 항상 화살표 커서) =====
st.markdown(
    """
    <style>
    div[data-baseweb="select"] { cursor: default !important; }
    div[data-baseweb="select"] * { cursor: default !important; }
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
    format_func=lambda x: {"protan": "Protanopia",
                           "deutan": "Deuteranopia",
                           "tritan": "Tritanopia"}[x],
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
        # EXIF 회전 보정 + RGB 고정
        uploaded_img = ImageOps.exif_transpose(Image.open(img_file)).convert("RGB")

with col_u2:
    st.subheader("② 사용 방법")
    st.markdown(
        "- 좌측에서 이미지를 업로드하세요.\n"
        "- 색각 유형, 해상도, 보정 강도를 사이드바에서 조정하세요.\n"
        "- 아래에서 원본/보정 결과를 나란히 비교하고 다운로드할 수 있습니다."
    )

st.divider()

if uploaded_img is None:
    st.info("좌측에서 이미지를 업로드해 주세요.")
    st.stop()

# ===== 처리 파이프라인 =====
# 1) 안전 리사이즈 -> PIL
pil_small = safe_resize(uploaded_img, target_long=max_width)

# 2) PIL -> OpenCV(BGR)
cv_small = pil_to_cv(pil_small)

# 3) ctype 문자열 정규화
def normalize_ctype(c: str) -> str:
    c = (c or "").lower()
    mapping = {
        "protan": "protanopia", "protanopia": "protanopia",
        "deutan": "deuteranopia", "deuteranopia": "deuteranopia",
        "tritan": "tritanopia", "tritanopia": "tritanopia",
    }
    return mapping.get(c, c)

ctype_norm = normalize_ctype(ctype)


# confusion-line 기반 보정 실행
kind = "protan" if ctype_norm.startswith("prot") else \
        "deutan" if ctype_norm.startswith("deut") else \
        "tritan"

# severity는 결함 시뮬 강도(0~1). 여기선 1.0(완전 결함 가정)로 고정하거나,
# 추후 사용자 입력으로 노출해도 됨.
corrected = daltonize_confusion_line_bgr(cv_small, kind=kind, alpha=alpha, severity=1.0)

# 변화량(디버깅)
diff = float(np.mean(np.abs(corrected.astype(np.int16) - cv_small.astype(np.int16))))
st.sidebar.write("보정 차이:", round(diff, 3))

# 변화량 표시
diff = float(np.mean(np.abs(corrected.astype(np.int16) - cv_small.astype(np.int16))))
st.sidebar.write("보정 차이:", round(diff, 3))


# 디버깅용 차이 값(수치가 0에 가깝다면 변화가 거의 없음)
diff = float(np.mean(np.abs(corrected.astype(np.int16) - cv_small.astype(np.int16))))
st.sidebar.write("보정 차이:", round(diff, 3))

# ===== 출력 & 다운로드 =====
masked_src = cv_small
masked_dst = corrected

c1, c2 = st.columns([1, 1], gap="medium")
src_pil = cv_to_pil(masked_src)
dst_pil = cv_to_pil(masked_dst)

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
    )

with c2:
    st.subheader("보정 결과")
    st.image(dst_pil, use_column_width=True)
    buf_dst = BytesIO()
    dst_pil.save(buf_dst, format="PNG")
    st.download_button(
        "✅ 보정 이미지 다운로드",
        data=buf_dst.getvalue(),
        file_name=f"truecolor_{ctype_norm}_alpha{alpha}_{max_width}px.png",
        mime="image/png",
    )

# 전/후 비교(병치) + 다운로드
st.subheader("전/후 비교 (가로 병치)")
compare_cv = side_by_side(masked_src, masked_dst, gap=16)
compare_pil = cv_to_pil(compare_cv)

c3, c4 = st.columns([1, 1], gap="medium")
with c3:
    st.image(src_pil, use_column_width=True, caption="원본")
with c4:
    st.image(dst_pil, use_column_width=True, caption=f"보정 ({ctype_norm}, α={alpha})")

comp_buf = BytesIO()
compare_pil.save(comp_buf, format="PNG")
st.download_button(
    "↔️ 전/후 비교(병치) 다운로드",
    data=comp_buf.getvalue(),
    file_name=f"truecolor_compare_{ctype_norm}_alpha{alpha}_{max_width}px.png",
    mime="image/png",
)

st.caption("Tip: 사이드바에서 해상도와 보정 강도를 조절해 성능/품질/효과를 맞춰보세요.")
