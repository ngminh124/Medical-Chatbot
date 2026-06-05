#!/bin/bash
set -e

PORT=${PORT:-7861}
HOST=${HOST:-0.0.0.0}
MODEL_PATH=${MODEL_PATH:-./models/qwen3-4b}

if ! command -v python >/dev/null 2>&1; then
	echo "python executable not found!"
	exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
	echo "GPU is not available. Ensure NVIDIA drivers are installed."
	exit 1
fi

if [ -n "${HF_TOKEN:-}" ]; then
	export HF_TOKEN
fi

if [ -n "${VLLM_API_KEY:-}" ]; then
	export VLLM_API_KEY
fi

echo "Starting vLLM..."
echo "Model: ${MODEL_PATH}"
echo "Host: ${HOST}"
echo "Port: ${PORT}"

exec python -m vllm.entrypoints.openai.api_server \
	--model "${MODEL_PATH}" \
	--host "${HOST}" \
	--port "${PORT}" \
	--gpu-memory-utilization 0.9 \
	--enable-prefix-caching \
	--max-model-len 8192 \
	--trust-remote-code
