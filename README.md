# TrueColor – 색각 이상자를 위한 색상 보정 웹앱

Python + OpenCV + Streamlit 기반의 웹앱으로, Protan/Deutan/Tritan 유형에 맞춰 **이미지 색상을 보정**해 색 구분을 돕습니다.

- **데모**: https://truecolor.onrender.com
- **보고서(요약본, 증빙용)**: (업로드 후 링크 삽입)
- **주요 기술**: Python, OpenCV, Streamlit, Render

## ✨ 기능
- 이미지 업로드 또는 브라우저 카메라 촬영
- 색각 유형 선택(Protan/Deutan/Tritan)
- 보정 전/후 **나란히 비교**(원형 마스크 포함)
- 경량 **채도/대비 보정**으로 구분력 향상

## 📸 스크린샷
<p align="center">
  <img src="docs/ui_main.png" width="45%"/>
  <img src="docs/compare_before_after.png" width="45%"/>
</p>

## ⚙️ 로컬 실행
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
