#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-qwen3:4b}"

echo "========================================="
echo " Installing Ollama"
echo "========================================="

if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed."
fi

echo
echo "========================================="
echo " Starting Ollama Server"
echo "========================================="

export OLLAMA_HOST="0.0.0.0:11434"

if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
    nohup ollama serve > ollama.log 2>&1 &
    sleep 5
else
    echo "Ollama server already running."
fi

echo
echo "========================================="
echo " Waiting for Ollama API"
echo "========================================="

until curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
    echo "Waiting..."
    sleep 2
done

echo "Ollama is ready."

echo
echo "========================================="
echo " Downloading Model: ${MODEL}"
echo "========================================="

ollama pull "${MODEL}"

echo
echo "========================================="
echo " Installed Models"
echo "========================================="

ollama list

echo
echo "========================================="
echo " Test Generation"
echo "========================================="

ollama run "${MODEL}" "Xin chào, hãy trả lời ngắn gọn rằng mô hình đã hoạt động."

echo
echo "========================================="
echo " Setup Complete"
echo "========================================="
echo "API Endpoint:"
echo "http://0.0.0.0:11434"
echo
echo "Test API:"
echo "curl http://localhost:11434/api/tags"
echo
echo "Generate:"
echo "curl http://localhost:11434/api/generate -d '{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"stream\":false}'"