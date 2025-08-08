import streamlit as st
from PIL import Image

def run_color_vision_test():
    st.subheader("🧠 간단한 색각 이상 테스트")

    score = 0

    # 1번 질문
    st.markdown("1️⃣ 아래 이미지에서 어떤 숫자가 보이나요?")
    img1 = Image.open("data/ishihara_images/plate_5.png")
    st.image(img1, width=200)
    ans1 = st.radio("숫자 선택", ["5", "12", "8", "못 보겠음"], key="q1")
    if ans1 == "5":
        score += 1

    # 2번 질문
    st.markdown("2️⃣ 다음 중 비슷해 보이는 색 조합을 선택하세요.")
    ans2 = st.radio("선택", [
        "빨강-초록", "파랑-노랑", "보라-검정"
    ], key="q2")
    if ans2 == "빨강-초록":
        score += 1

    # 결과
    st.markdown("---")
    if score == 2:
        st.success("정상 또는 약한 색각 이상 가능성 있음")
        return "normal"
    elif ans2 == "빨강-초록":
        st.warning("Protanopia 또는 Deuteranopia 가능성")
        return "red-green"
    elif ans2 == "파랑-노랑":
        st.warning("Tritanopia 가능성")
        return "tritan"
    else:
        st.info("정확한 검사가 필요합니다.")
        return "unknown"
