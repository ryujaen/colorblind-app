import streamlit as st
from PIL import Image
import numpy as np
import cv2
import test_cvd
from color_filter import apply_colorblind_filter

st.set_page_config(page_title="색각이상자 색상 보정 앱")

st.title("🎨 색각이상자 색상 보정 앱")
st.write("사진을 업로드하고 색각 이상 유형을 선택하면, 변환된 이미지를 보여줍니다.")

know_type = st.selectbox("색각 이상 유형을 알고 계신가요?", ["예", "아니요"])

if know_type == "아니요":
    user_type = test_cvd.run_color_vision_test()  # → 결과: "Protanopia", "Deuteranopia", 등
    color_type = user_type
else:
    color_type = st.selectbox(
        "색각 이상 유형 선택",
        ["Protanopia (적색맹)", "Deuteranopia (녹색맹)", "Tritanopia (청색맹)"],
        key="color_type_select"
    )

uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])
color_type = st.selectbox(
    "색각 이상 유형 선택",
    ["Protanopia (적색맹)", "Deuteranopia (녹색맹)", "Tritanopia (청색맹)"]
)

if uploaded_file:
    # 이미지 열기
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 필터 적용
    filtered = apply_colorblind_filter(img_cv, color_type)
    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    # 결과 출력
    st.subheader("원본 이미지")
    st.image(image, use_column_width=True)

    st.subheader("보정된 이미지")
    st.image(filtered_rgb, use_column_width=True)
