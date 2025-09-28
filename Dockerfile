# ---- Base Python image
FROM python:3.11-slim

# ---- System libraries (OpenCV 실행에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ---- 작업 디렉토리 설정
WORKDIR /app

# ---- 의존성 먼저 설치 (캐시 활용)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# ---- 앱 소스 복사
COPY . /app

# ---- 환경 변수
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ---- 컨테이너 실행 시 실행할 명령어
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
