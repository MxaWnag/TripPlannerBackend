#!/usr/bin/env bash
# 关闭全部：关闭 FastAPI (uvicorn :8001) + Docker 服务
# 启动全部: ./start.sh （开启 FastAPI + Docker）
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 关闭 FastAPI ----------
echo "[stop.sh] 关闭 FastAPI (uvicorn :8001)..."
if [ -f ".uvicorn.pid" ]; then
  PID=$(cat .uvicorn.pid)
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    echo "[stop.sh] FastAPI 已关闭，已结束进程 $PID"
  fi
  rm -f .uvicorn.pid
else
  echo "[stop.sh] 未找到 .uvicorn.pid，尝试按端口 8001 结束进程..."
  (lsof -ti :8001 | xargs -r kill 2>/dev/null) || true
fi
# ---------- FastAPI 关闭结束 ----------

echo "[stop.sh] 关闭 Docker 服务..."
docker compose down

echo "全部已关闭。（FastAPI + Docker）"
