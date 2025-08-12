import streamlit as st
from PIL import Image
import numpy as np
import cv2
import test_cvd

from color_filter import apply_colorblind_filter
from color_filter import simulate_cvd_rgb  # ← 방금 만든 함수

st.set_page_config(page_title="색각이상자 색상 보정 앱")
st.title("🎨 색각이상자 색상 보정 앱")
st.write("사진을 업로드하거나 색각 이상 유형을 선택하면, 변환된 이미지를 보여줍니다.")

know_type = st.selectbox("색각 이상 유형을 알고 계신가요?", ["예", "아니요"])

# 색각 이상 유형 질문
if know_type == "아니요":
    cvd_key, auto_sev = test_cvd.run_color_vision_test()  # cvd_key: 'protanomaly' | 'deuteranomaly' | 'tritanomaly' | 'normal'
    # 테스트가 끝난 경우에만 기본값 자동 설정
    if cvd_key:
        st.session_state["cvd_type"] = cvd_key
        st.session_state["cvd_severity"] = auto_sev
        # 필터용 드롭다운(한국어 라벨) 기본 선택을 매핑
        label_map = {
            "protanomaly": "Protanopia (적색맹)",
            "deuteranomaly": "Deuteranopia (녹색맹)",
            "tritanomaly": "Tritanopia (청색맹)",
            "normal": "Deuteranopia (녹색맹)"  # 기본값 아무거나; normal이면 필터 안씀
        }
        color_type = label_map.get(cvd_key, "Deuteranopia (녹색맹)")
    else:
        color_type = st.session_state.get("color_type_select", "Deuteranopia (녹색맹)")
else:
    color_type = st.selectbox(
        "색각 이상 유형 선택",
        ["Protanopia (적색맹)", "Deuteranopia (녹색맹)", "Tritanopia (청색맹)"],
        key="color_type_select"
    )
# 이미지 입력 방식 선택
st.write("📤 이미지를 어떻게 불러올까요?")
input_method = st.radio("이미지 선택 방식", ["파일 업로드"])

uploaded_file = None
camera_photo = None
image = None

# 업로드 또는 카메라로 이미지 가져오기
if input_method == "파일 업로드":
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
'''
elif input_method == "카메라로 사진 찍기":
    camera_photo = st.camera_input("카메라에서 프레임이 보이면 캡처 버튼을 눌러주세요.")
    if camera_photo:
        image = Image.open(camera_photo).convert("RGB")
'''
# 이미지가 준비되었으면 처리
if image is not None:
    st.subheader("🖼 원본 이미지")
    st.image(image, caption="입력 이미지", use_column_width=True)

    # OpenCV 변환
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 색각이상 필터 적용
    filtered = apply_colorblind_filter(img_cv, color_type)
    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    st.subheader("🔧 보정된 이미지")
    st.image(filtered_rgb, use_column_width=True)

st.info("현재 웹 배포 환경에서는 카메라 기능이 제한되어 이미지 업로드만 지원됩니다.")


st.markdown("### 👀 색각 이상 시뮬레이션 보기")
sim_on = st.checkbox("색각 이상자의 시선에서 보기 (원본/보정 모두)")

if sim_on:
    sim_type = st.selectbox(
        "시뮬레이션 유형",
        ["Protanopia (protanomaly)", "Deuteranopia (deuteranomaly)", "Tritanopia (tritanomaly)"]
    )
    severity = st.slider("시뮬레이션 강도(severity)", 0, 100, 100, 5)

    # 문자열 매핑
    map_key = {
        "Protanopia (protanomaly)": "protanomaly",
        "Deuteranopia (deuteranomaly)": "deuteranomaly",
        "Tritanopia (tritanomaly)": "tritanomaly",
    }
    cvd_key = map_key[sim_type]

    # 원본/보정 각각 시뮬레이션
    orig_rgb = np.array(image.convert("RGB"))

    filtered_rgb = cv2.cvtColor(filtered_rgb, cv2.COLOR_BGR2RGB)

    orig_sim = simulate_cvd_rgb(orig_rgb, cvd_key, severity=severity)
    filt_sim = simulate_cvd_rgb(filtered_rgb, cvd_key, severity=severity)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("원본 (시뮬레이션)")
        st.image(orig_sim, use_container_width=True)
    with c2:
        st.caption("보정본 (시뮬레이션)")
        st.image(filt_sim, use_container_width=True)