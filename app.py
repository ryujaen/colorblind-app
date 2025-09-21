# app.py — TrueColor (정리/보강 최종본)
from io import BytesIO
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps

from daltonize import correct_image
from image_utils import pil_to_cv, cv_to_pil, safe_resize, side_by_side

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

# 4) 보정 실행 (색공간/시그니처 자동 호환)
def run_correct(img_bgr: np.ndarray, user_ctype: str, alpha_val: float) -> np.ndarray:
    """
    - 입력: BGR uint8 [0..255]
    - 내부: RGB float [0..1] 로 변환 후 daltonize.correct_image 호출
    - 출력: BGR uint8 [0..255]
    """
    # 후보 토큰 정규화
    base = (user_ctype or "").lower()
    family = {
        "protan": ["protan", "protanopia", "p"],
        "deutan": ["deutan", "deuteranopia", "d"],
        "tritan": ["tritan", "tritanopia", "t"],
    }
    key = "protan" if base.startswith("prot") else \
          "deutan" if base.startswith("deut") else \
          "tritan" if base.startswith("trit") else base
    tokens = family.get(key, [user_ctype])

    # BGR uint8 -> RGB float
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _try(token):
        # 일부 구현은 alpha 인자를 지원 안 하므로 두 번 시도
        try:
            out = correct_image(rgb, ctype=token, alpha=alpha_val)
        except TypeError:
            out = correct_image(rgb, ctype=token)
        return out

    best = None
    best_diff = -1.0
    for tk in tokens:
        try:
            out_rgb = _try(tk)
            if isinstance(out_rgb, np.ndarray):
                # 결과가 0..1/0..255/float/uint8 등 섞여 있을 수 있으니 정규화
                o = out_rgb.astype(np.float32)
                if o.max() > 1.01:   # 0..255 범위로 온 경우
                    o = o / 255.0
                o = np.clip(o, 0.0, 1.0)

                # 다시 BGR uint8로
                out_bgr = cv2.cvtColor((o * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

                # 차이가 큰 쪽을 채택(실제 변화가 있었는지 확인)
                diff = float(np.mean(np.abs(out_bgr.astype(np.int16) - img_bgr.astype(np.int16))))
                if diff > best_diff:
                    best, best_diff = out_bgr, diff
        except Exception:
            continue

    # 실패하면 원본 반환
    return best if isinstance(best, np.ndarray) else img_bgr

# 보정 실행 (라이브러리에 alpha 전달)
corrected = run_correct(cv_small, ctype_norm, alpha)

# 디버깅용 차이 값
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
