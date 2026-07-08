#!/bin/bash
# Double-click (or ./Launch_Linux.sh) launcher for Enterprise Knowledge Assistant (Linux)

cd "$(dirname "$0")"
echo "=== Enterprise Offline Knowledge Assistant ==="
echo ""

# --- Start Ollama if not running ---
if ! curl -s http://127.0.0.1:11434 > /dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
else
    echo "Ollama already running."
fi

# --- Ensure models are pulled ---
if ! ollama list | grep -q "llama3.1:8b"; then
    echo "Pulling llama3.1:8b (one-time, ~4.9GB)..."
    ollama pull llama3.1:8b
fi
if ! ollama list | grep -q "nomic-embed-text"; then
    echo "Pulling nomic-embed-text (one-time)..."
    ollama pull nomic-embed-text
fi

# --- Ensure Python packages are installed ---
echo "Checking Python packages..."
pip3 install --quiet -r requirements.txt

# --- Knowledge base building now happens inside the app itself, per
# selected knowledge base under KnowledgeBases/<name>/ — nothing to
# pre-check here anymore.

# --- Launch the app ---
echo ""
echo "Launching app in your browser..."
streamlit run app.py

echo ""
read -n 1 -p "App closed. Press any key to close this window..."
