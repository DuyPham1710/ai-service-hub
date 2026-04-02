FROM python:3.10-slim

WORKDIR /app

# Khắc phục lỗi build có thể xảy ra trong một số môi trường (Option)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện trước để tận dụng Docker Cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào image
COPY . .

# Port mặc định của FastAPI
EXPOSE 8000

# Lệnh khởi chạy server
CMD ["python", "main.py"]
