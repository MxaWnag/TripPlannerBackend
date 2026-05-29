#!/usr/bin/env bash
# 关闭：FastAPI + 卸载 Agent 模型 + Docker；若 start.sh 曾拉起本机 ollama serve 则一并停止
# 启动: ./start.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

AGENT_LLM_MODEL="${AGENT_LLM_MODEL:-qwen2.5:14b-instruct}"
if [ -f ".ollama.model" ]; then
  AGENT_LLM_MODEL="$(tr -d '[:space:]' < .ollama.model)"
fi

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

# ---------- 卸载 Agent 模型（释放显存）----------
echo "[stop.sh] 卸载 Agent 模型 (${AGENT_LLM_MODEL})..."
if command -v ollama >/dev/null 2>&1; then
  if ollama ps 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$AGENT_LLM_MODEL"; then
    if ollama stop "$AGENT_LLM_MODEL" 2>/dev/null; then
      echo "[stop.sh] 已从显存卸载: ${AGENT_LLM_MODEL}"
    else
      echo "[stop.sh] 警告: ollama stop ${AGENT_LLM_MODEL} 失败"
    fi
  else
    echo "[stop.sh] 模型未在显存中，跳过 ollama stop"
  fi
else
  echo "[stop.sh] 未找到 ollama 命令，跳过模型卸载"
fi
rm -f .ollama.model
# ---------- 模型卸载结束 ----------

# ---------- 关闭由 start.sh 拉起的本机 Ollama ----------
if [ -f ".ollama.pid" ]; then
  PID=$(cat .ollama.pid)
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    echo "[stop.sh] 已停止由 start.sh 启动的本机 Ollama (PID $PID)"
  fi
  rm -f .ollama.pid
fi
# 若 Ollama 由系统/桌面自行管理，此处不会停止

echo "[stop.sh] 关闭 Docker 服务..."
docker compose down

echo "全部已关闭。（FastAPI + Agent 模型显存 + Docker；未停止你自行管理的本机 ollama serve 进程）"
