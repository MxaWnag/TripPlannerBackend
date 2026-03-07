#!/usr/bin/env bash
# 启动全部服务：Docker（qdrant, ollama, db, web）+ 开启 FastAPI (uvicorn :8001)
# 关闭全部: ./stop.sh （关闭 FastAPI + Docker）
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[start.sh] 启动 Docker 服务 (qdrant, ollama, db, web)..."
docker compose up -d

echo "[start.sh] 等待服务就绪..."
sleep 5

# ---------- 开启 FastAPI ----------
if [ -d ".venv" ]; then
  echo "[start.sh] 使用 .venv"
  source .venv/bin/activate
fi

if [ -f ".uvicorn.pid" ]; then
  OLD_PID=$(cat .uvicorn.pid)
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[start.sh] FastAPI 已在运行 (PID $OLD_PID)，跳过开启"
  else
    rm -f .uvicorn.pid
  fi
fi

if [ ! -f ".uvicorn.pid" ] || ! kill -0 "$(cat .uvicorn.pid)" 2>/dev/null; then
  echo "[start.sh] 开启 FastAPI: uvicorn fast_api_embedded_server:app --host 0.0.0.0 --port 8001"
  nohup uvicorn fast_api_embedded_server:app --host 0.0.0.0 --port 8001 > .uvicorn.log 2>&1 &
  echo $! > .uvicorn.pid
  echo "[start.sh] FastAPI 已开启，PID: $(cat .uvicorn.pid)，日志: .uvicorn.log"
fi
# ---------- FastAPI 开启结束 ----------

echo ""
echo "全部启动完成。"
echo "  Docker:  qdrant :6333, ollama :11434, db :5432, web :8000"
echo "  FastAPI: http://localhost:8001  (docs: http://localhost:8001/docs)"
echo "  关闭全部（关闭 FastAPI + Docker）: ./stop.sh"
