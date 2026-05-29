FROM python:3.11-slim

WORKDIR /app

# 可选：工具链（用 psycopg2-binary 时通常不需要，但留着更稳）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# 代码在 compose 里用挂载，镜像里也放一份以便构建成功
COPY . .

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
