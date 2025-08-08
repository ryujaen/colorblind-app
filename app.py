import streamlit as st
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
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

st.write("📤 이미지를 어떻게 불러올까요?")
input_method = st.radio("이미지 선택 방식", ["파일 업로드", "카메라로 사진 찍기"])

uploaded_file = None
captured_image = None

# 카메라 입력 처리 클래스 정의
class CameraTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")
        self.frame = image
        return image

# 3. 이미지 로드
if input_method == "파일 업로드":
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])
elif input_method == "카메라로 사진 찍기":
    st.warning("카메라에서 프레임이 보이면 캡처 버튼을 눌러주세요.")
    ctx = webrtc_streamer(key="camera", video_transformer_factory=CameraTransformer)
    if ctx.video_transformer and ctx.video_transformer.frame is not None:
        if st.button("📸 사진 캡처"):
            captured_image = ctx.video_transformer.frame

# 4. 이미지 처리
image = None


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif captured_image is not None:
    image = Image.fromarray(cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB))

if image is not None:
    # PIL → NumPy → OpenCV
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 필터 적용
    filtered = apply_colorblind_filter(img_cv, color_type)
    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    # 결과 출력
    st.subheader("🖼️ 원본 이미지")
    st.image(image, use_column_width=True)

    st.subheader("🎯 보정된 이미지")
    st.image(filtered_rgb, use_column_width=True)