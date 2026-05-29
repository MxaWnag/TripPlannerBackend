#!/usr/bin/env bash
# 启动：本机 Ollama + Docker（qdrant, db, web）+ FastAPI (uvicorn :8001)
# 关闭: ./stop.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
OLLAMA_HOST="${OLLAMA_URL#*://}"
OLLAMA_HOST="${OLLAMA_HOST%%/*}"

AGENT_LLM_MODEL="${AGENT_LLM_MODEL:-qwen2.5:14b-instruct}"
OLLAMA_WARMUP_TIMEOUT="${OLLAMA_WARMUP_TIMEOUT:-180}"

ollama_ready() {
  curl -sf "${OLLAMA_URL%/}/api/version" >/dev/null 2>&1
}

ollama_model_pulled() {
  local model="$1"
  command -v ollama >/dev/null 2>&1 || return 1
  ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$model"
}

ollama_model_loaded() {
  local model="$1"
  command -v ollama >/dev/null 2>&1 || return 1
  ollama ps 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$model"
}

warmup_ollama_model() {
  local model="$1"
  echo "[start.sh] Agent 模型 (${model})..."

  if ! command -v ollama >/dev/null 2>&1; then
    echo "[start.sh] 警告: 未找到 ollama 命令，跳过模型检测/预热"
    return 0
  fi

  if ! ollama_model_pulled "$model"; then
    echo "[start.sh] 本地未安装 ${model}，执行 ollama pull..."
    if ! ollama pull "$model"; then
      echo "[start.sh] 错误: ollama pull ${model} 失败"
      exit 1
    fi
  else
    echo "[start.sh] 模型已安装: ${model}"
  fi

  if ollama_model_loaded "$model"; then
    echo "[start.sh] 模型已在显存中: ${model}"
    echo "$model" > .ollama.model
    ollama ps 2>/dev/null | sed -n '1,3p' || true
    return 0
  fi

  echo "[start.sh] 预热加载模型（首次约 1–2 分钟，最长等待 ${OLLAMA_WARMUP_TIMEOUT}s）..."
  if ! curl -sf "${OLLAMA_URL%/}/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model}\",\"prompt\":\"OK\",\"stream\":false,\"options\":{\"num_predict\":8}}" \
    --max-time "$OLLAMA_WARMUP_TIMEOUT" >/dev/null; then
    echo "[start.sh] 错误: 模型 ${model} 预热失败（超时或 Ollama 报错），请查看 .ollama.log"
    exit 1
  fi

  if ! ollama_model_loaded "$model"; then
    echo "[start.sh] 警告: 预热请求已完成，但 ollama ps 未显示 ${model}（可能已自动卸载）"
  else
    echo "[start.sh] 模型已加载到显存"
    ollama ps 2>/dev/null | sed -n '1,3p' || true
  fi
  echo "$model" > .ollama.model
}

# 释放 11434：旧版 compose 可能仍留着 ollama 容器占用端口
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx 'ollama'; then
  echo "[start.sh] 停止并移除 Docker 容器 ollama（改用本机 Ollama）..."
  docker stop ollama >/dev/null 2>&1 || true
  docker rm ollama >/dev/null 2>&1 || true
fi

echo "[start.sh] 检查本机 Ollama (${OLLAMA_URL})..."
if ollama_ready; then
  echo "[start.sh] 本机 Ollama 已就绪"
elif command -v ollama >/dev/null 2>&1; then
  echo "[start.sh] 启动本机 Ollama: ollama serve"
  nohup ollama serve > .ollama.log 2>&1 &
  echo $! > .ollama.pid
  for _ in $(seq 1 30); do
    if ollama_ready; then
      echo "[start.sh] 本机 Ollama 已就绪 (PID $(cat .ollama.pid))"
      break
    fi
    sleep 1
  done
  if ! ollama_ready; then
    echo "[start.sh] 错误: 本机 Ollama 未在 ${OLLAMA_URL} 响应，请查看 .ollama.log 或手动运行: ollama serve"
    exit 1
  fi
else
  echo "[start.sh] 错误: 未找到 ollama 命令，且 ${OLLAMA_URL} 不可达。请先安装并启动本机 Ollama。"
  exit 1
fi

warmup_ollama_model "$AGENT_LLM_MODEL"

echo "[start.sh] 启动 Docker 服务 (qdrant, db, web)..."
docker compose up -d

echo "[start.sh] 等待服务就绪..."
sleep 5

# ---------- 开启 FastAPI ----------
if [ -d ".venv" ]; then
  echo "[start.sh] 使用 .venv"
  # shellcheck source=/dev/null
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
echo "  本机 Ollama: ${OLLAMA_URL}"
echo "  Agent 模型:  ${AGENT_LLM_MODEL}  (ollama ps 查看显存占用)"
echo "  Docker:      qdrant :6333, db :5433, web :8000"
echo "  FastAPI:     http://localhost:8001  (docs: http://localhost:8001/docs)"
echo "  关闭:        ./stop.sh  （停 FastAPI、卸载模型、Docker；本脚本拉起的 ollama serve 也会停）"
