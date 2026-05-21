#!/bin/bash
set -e

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

COLOR_RESET="\033[0m"
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[0;33m"
COLOR_RED="\033[0;31m"
COLOR_BLUE="\033[0;34m"

log_info() {
  echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $*"
}

log_warn() {
  echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $*"
}

log_error() {
  echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $*"
}

log_ok() {
  echo -e "${COLOR_GREEN}[OK]${COLOR_RESET} $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log_error "Missing required command: $1"
    exit 1
  fi
}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B}
VLLM_PORT=${VLLM_PORT:-8001}
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}"

VLLM_PID=""
CF_PID=""
CF_LOG=""

cleanup() {
  log_warn "Shutting down..."
  if [ -n "$CF_PID" ] && kill -0 "$CF_PID" >/dev/null 2>&1; then
    log_info "Stopping cloudflared (PID: $CF_PID)"
    kill "$CF_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    log_info "Stopping vLLM (PID: $VLLM_PID)"
    kill "$VLLM_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup INT TERM EXIT

check_vllm_health() {
  curl -s "${VLLM_BASE_URL}/v1/models" >/dev/null 2>&1
}

wait_for_vllm() {
  local start_ts
  start_ts=$(date +%s)
  local timeout_seconds=300
  log_info "Waiting for vLLM at ${VLLM_BASE_URL}/v1/models (timeout ${timeout_seconds}s)"
  while true; do
    if check_vllm_health; then
      log_ok "vLLM is ready"
      return 0
    fi
    local now
    now=$(date +%s)
    if [ $((now - start_ts)) -ge $timeout_seconds ]; then
      log_error "vLLM did not become ready within ${timeout_seconds}s"
      return 1
    fi
    sleep 2
  done
}

start_cloudflared() {
  if ! command -v cloudflared >/dev/null 2>&1; then
    log_warn "cloudflared not found; skipping tunnel"
    return 0
  fi

  CF_LOG=$(mktemp)
  log_info "Starting cloudflared tunnel for http://localhost:7860"
  cloudflared tunnel --url http://localhost:7860 --no-autoupdate >"$CF_LOG" 2>&1 &
  CF_PID=$!
  log_info "cloudflared PID: $CF_PID"

  local i
  for i in $(seq 1 120); do
    local url
    url=$(grep -oE 'https://[^ ]+trycloudflare.com' "$CF_LOG" | tail -n 1)
    if [ -n "$url" ]; then
      log_ok "cloudflared public URL: $url"
      return 0
    fi
    if ! kill -0 "$CF_PID" >/dev/null 2>&1; then
      log_warn "cloudflared exited before URL was detected"
      return 1
    fi
    sleep 1
  done
  log_warn "cloudflared URL not detected after 120s"
  return 1
}

main() {
  require_cmd python3
  require_cmd curl

  log_info "Project root: $ROOT_DIR"
  log_info "MODEL_NAME=${MODEL_NAME}"
  log_info "VLLM_PORT=${VLLM_PORT}"
  log_info "VLLM_BASE_URL=${VLLM_BASE_URL}"

  log_info "Starting vLLM..."
  python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code &

  VLLM_PID=$!
  log_info "vLLM PID: ${VLLM_PID}"

  wait_for_vllm

  export VLLM_BASE_URL

  start_cloudflared || true

  log_info "Starting Qwen3 model service (foreground)"
  python3 "$ROOT_DIR/serving/qwen3_models/app.py"
}

main "$@"
