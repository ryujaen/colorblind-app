# ---- Base image
FROM python:3.11-slim

# ---- System libs needed by OpenCV (no GUI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ---- Workdir
WORKDIR /app

# ---- Install deps first (cache-friendly)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy app source
COPY . /app

# ---- Env
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

# ---- Run (MUST use $PORT for Vercel)
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0
