#!/bin/bash
set -e

export HF_TOKEN=$HF_TOKEN
export VLLM_API_KEY=$VLLM_API_KEY

PORT=7861
HOST=0.0.0.0

# Check uv installation
UV_CMD=$(which uv 2>/dev/null)
if [ -z "$UV_CMD" ]; then
	echo "uv executable not found!"
	exit 1
fi

echo "Using uv executable: $UV_CMD"

# Start vLLM
echo "Starting vLLM..."
exec uv run -m vllm.entrypoints.openai.api_server \
	--model Qwen/Qwen3-4B \
	--host ${HOST} \
	--port ${PORT} \
	--gpu-memory-utilization 0.9 \
	--enable-prefix-caching \
	--max-model-len 8192 \
	--trust-remote-code \
	--device cuda
