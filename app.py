import streamlit as st
from PIL import Image
import numpy as np
import cv2
import test_cvd
from color_filter import apply_colorblind_filter


st.set_page_config(page_title="색각이상자 색상 보정 앱")
st.title("🎨 색각이상자 색상 보정 앱")
st.write("사진을 업로드하거나 색각 이상 유형을 선택하면, 변환된 이미지를 보여줍니다.")

# 색각 이상 유형 질문
know_type = st.selectbox("색각 이상 유형을 알고 계신가요?", ["예", "아니요"])

if know_type == "아니요":
    user_type = test_cvd.run_color_vision_test()
    color_type = user_type
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
