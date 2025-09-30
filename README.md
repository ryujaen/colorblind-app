# TrueColor – 색각 이상자를 위한 색상 보정 웹앱

Python + OpenCV + Streamlit 기반의 웹앱으로, Protan/Deutan/Tritan 유형에 맞춰 **이미지 색상을 보정**해 색 구분을 돕습니다.  
Daltonization(일반 시야 가독성 강화)과 Inverse Compensation(색각 시야에서 원본과 유사화) 두 모드를 제공합니다.

- **데모**: https://truecolor.onrender.com  _(Render · Free plan · cold start 지연 가능)_
- **보고서(요약본, 증빙용)**: _(업로드 후 링크 삽입)_
- **주요 기술**: Python, OpenCV, NumPy, Pillow, Streamlit, Render

---

## ✨ 주요 기능

- **입력**: 이미지 업로드(JPG/PNG) 또는 **브라우저 카메라 촬영**
- **모드**:
  - *Daltonization* — 일반 시야에서 색 구분/가독성 강화
  - *Inverse Compensation* — 색각 시야에서 원본과 유사하게 보이도록 역보정
- **색각 유형 선택**: Protan / Deutan / Tritan
- **파라미터 조절**:
  - 보정 강도 **α** (0.0–1.0)
  - 결함 강도 **severity** (0.0–1.0)
  - **출력 해상도** (긴 변 기준 px)
- **결과 확인**:
  - **보정 전·후 나란히 비교(병치)** + 원형 마스크 미리보기
  - **색각 시야 시뮬레이션(전/후)** 제공
- **정량 지표**: **시야 오차(MAE, ↓좋음)** 표시로 보정 품질 수치 확인
- **다운로드**: 원본 / 보정 / 전후 비교 이미지를 각각 저장

---

## 📸 스크린샷

<p align="center">
  <img src="docs/ui_main.png" width="45%" alt="UI 메인 화면"/>
  <img src="docs/compare_before_after.png" width="45%" alt="보정 전/후 병치 비교"/>
</p>

<p align="center">
  <img src="docs/simulation.png" width="45%" alt="색각 시야 시뮬레이션"/>
  <img src="docs/download.png" width="45%" alt="전후 비교 이미지 다운로드"/>
</p>

> 스크린샷 파일이 없다면 `docs/` 폴더에 PNG를 추가하고 위 경로를 맞춰 주세요.

---

## 🧪 로컬 실행

```bash
# 1) 가상환경
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 실행
streamlit run app.py
