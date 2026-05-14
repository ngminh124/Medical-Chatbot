#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅${NC} $1"; }
warn() { echo -e "${YELLOW}⚠️${NC}  $1"; }
fail() { echo -e "${RED}❌${NC} $1"; }

check_url() {
  local name="$1"; local url="$2"
  local code
  code=$(curl -sS -m 4 -o /tmp/health.out -w '%{http_code}' "$url" || true)
  if [[ "$code" == "200" ]]; then
    pass "$name ($url)"
  else
    fail "$name ($url) -> HTTP $code"
  fi
}

echo "==== Minqes stack check ===="

echo "[1/7] Process check"
ps -ef | grep -E 'serving/qwen3_models/app.py|vllm.entrypoints.openai.api_server|uvicorn backend.main:app|ollama serve' | grep -v grep || true

echo

echo "[2/7] Port check"
ss -ltnp | grep -E ':7860|:7861|:8000|:11434|:9200|:6379|:6333|:5432' || true

echo

echo "[3/7] Endpoint health"
check_url "Backend health" "http://localhost:8000/v1/health"
check_url "Qwen3 ready (embed/rerank/guard)" "http://localhost:7860/v1/ready"
check_url "vLLM health" "http://localhost:7861/health"
check_url "Ollama tags" "http://localhost:11434/api/tags"
check_url "Elasticsearch" "http://localhost:9200/_cluster/health"
check_url "Qdrant root" "http://localhost:6333/"
check_url "STT health" "http://localhost:8000/v1/stt/health"
check_url "TTS health" "http://localhost:8000/v1/tts/health"

echo

echo "[4/7] Redis"
if redis-cli -h 127.0.0.1 -p 6379 ping >/tmp/redis_ping.out 2>&1; then
  if grep -q PONG /tmp/redis_ping.out; then
    pass "Redis PING"
  else
    warn "Redis reachable but ping is not PONG"
  fi
else
  fail "Redis not reachable on 127.0.0.1:6379"
fi

echo

echo "[5/7] Docker services"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true

echo

echo "[6/7] GPU/RAM capacity"
free -h || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
fi

echo

echo "[7/7] Quick diagnosis"
if curl -sS -m 2 http://localhost:7860/v1/ready >/dev/null 2>&1; then
  pass "Rerank + Guardrails service is UP via qwen3_models (:7860)"
else
  fail "Rerank + Guardrails service is DOWN -> start: /venv/bin/python serving/qwen3_models/app.py"
fi

if curl -sS -m 2 http://localhost:7861/health >/dev/null 2>&1; then
  pass "vLLM is UP (:7861)"
else
  warn "vLLM is DOWN (:7861). System will fallback to Ollama if configured."
fi

echo "==== Done ===="
