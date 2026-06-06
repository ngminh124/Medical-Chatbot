#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-qwen3:8b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-11434}"

echo "========================================="
echo " Ollama Startup"
echo "========================================="

# Check GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found. Ollama may run on CPU."
else
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Install Ollama if missing
if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed."
fi

export OLLAMA_HOST="${HOST}:${PORT}"

echo
echo "========================================="
echo " Starting Ollama Server"
echo "========================================="

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

cleanup() {
    echo "Stopping Ollama..."
    kill ${OLLAMA_PID} 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo
echo "========================================="
echo " Waiting for Ollama API"
echo "========================================="

until curl -fs "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; do
    echo "Waiting for Ollama..."
    sleep 2
done

echo "Ollama API ready."

echo
echo "========================================="
echo " Downloading Model"
echo "========================================="

if ! ollama list | awk '{print $1}' | grep -qx "${MODEL}"; then
    echo "Pulling ${MODEL} ..."
    ollama pull "${MODEL}"
else
    echo "Model ${MODEL} already exists."
fi

echo
echo "========================================="
echo " Installed Models"
echo "========================================="

ollama list

echo
echo "========================================="
echo " Ollama Ready"
echo "========================================="

echo "Model : ${MODEL}"
echo "Host  : ${HOST}"
echo "Port  : ${PORT}"

echo
echo "API Endpoint:"
echo "http://${HOST}:${PORT}"

echo
echo "Health Check:"
echo "curl http://localhost:${PORT}/api/tags"

echo
echo "Generate Example:"
echo "curl -X POST http://localhost:${PORT}/api/generate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"stream\":false}'"

echo
echo "========================================="
echo " Ollama Server Running"
echo "========================================="

# Giữ container/process sống
wait ${OLLAMA_PID}