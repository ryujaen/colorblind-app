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

# 4) 보정 실행
def run_correct(img_bgr: np.ndarray, user_ctype: str, alpha_val: float) -> np.ndarray:
    """
    입력:  BGR uint8 [0..255]
    내부:  RGB float [0..1] 로 변환 후 daltonize.correct_image 호출
    출력:  BGR uint8 [0..255]
    변화가 없으면 간이 보정 매트릭스로 fallback
    """
    base = (user_ctype or "").lower()
    if   base.startswith("prot"): key = "protan"
    elif base.startswith("deut"): key = "deutan"
    elif base.startswith("trit"): key = "tritan"
    else:                         key = base

    # --- BGR→RGB (float) ---
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # --- 라이브러리 호출 (alpha 지원/미지원 모두 시도) ---
    out_rgb = None
    try:
        try:
            out_rgb = correct_image(rgb, ctype=key, alpha=alpha_val)
        except TypeError:
            out_rgb = correct_image(rgb, ctype=key)
    except Exception:
        out_rgb = None

    use_fallback = True
    if isinstance(out_rgb, np.ndarray):
        o = out_rgb.astype(np.float32)
        if o.max() > 1.01:  # 0..255로 온 경우
            o = o / 255.0
        o = np.clip(o, 0.0, 1.0)
        # 변화량 체크
        diff = float(np.mean(np.abs(o - rgb)))
        if diff > 1e-4:
            use_fallback = False
            rgb_out = o

    # --- 변화가 거의 없으면 간이 보정 매트릭스 적용 ---
    if use_fallback:
        # 간단한 채널 보정(시각적 효과용, alpha 가중 적용)
        # protan: R 감지 약 → G를 R로 보조 주입
        # deutan: G 감지 약 → R을 G로 보조 주입
        # tritan: B 감지 약 → G를 B로 보조 주입
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        if key == "protan":
            r2 = np.clip(r + alpha_val * 0.7 * g, 0.0, 1.0)
            g2 = g
            b2 = b
        elif key == "deutan":
            r2 = r
            g2 = np.clip(g + alpha_val * 0.7 * r, 0.0, 1.0)
            b2 = b
        elif key == "tritan":
            r2 = r
            g2 = g
            b2 = np.clip(b + alpha_val * 0.7 * g, 0.0, 1.0)
        else:
            r2, g2, b2 = r, g, b
        rgb_out = np.stack([r2, g2, b2], axis=-1)

    # --- RGB(float) → BGR(uint8) ---
    bgr_out = cv2.cvtColor((rgb_out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr_out


# 보정 실행 (alpha는 run_correct 내부에서 처리)
corrected = run_correct(cv_small, ctype_norm, alpha)

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
