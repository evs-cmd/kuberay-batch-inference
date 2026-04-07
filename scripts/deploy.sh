#!/bin/bash
# deploy.sh — Deploy Ray + vLLM + FastAPI on a single GPU machine
#
# Usage:
#   cd ~/kuberay-batch-inference
#   bash scripts/deploy.sh
#
# Stops everything:
#   bash scripts/deploy.sh stop
set -euo pipefail

LOG="[deploy]"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/venv"
false

# ── Stop mode ────────────────────────────────────────────────────
if [ "${1:-}" = "stop" ]; then
    echo "$LOG Stopping..."
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    echo "$LOG Stopped."
    exit 0
fi

echo "$LOG ════════════════════════════════════════"
echo "$LOG  KubeRay Batch Inference — Deploy"
echo "$LOG ════════════════════════════════════════"

# ── GPU check ────────────────────────────────────────────────────
echo "$LOG Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Venv + deps ──────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "$LOG Creating venv at $VENV..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo "$LOG Installing dependencies..."
pip install --upgrade pip -q
pip install -q -r "$PROJECT_DIR/requirements/gpu.txt"

# ── Download + verify model ──────────────────────────────────────
echo "$LOG Downloading and verifying model..."
cd "$PROJECT_DIR"
python -m app.model_server

# ── Start Ray ────────────────────────────────────────────────────
ray stop --force 2>/dev/null || true
sleep 2

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "$LOG Starting Ray head ($GPU_COUNT GPU)..."

ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-cpus=$(nproc) \
    --num-gpus=$GPU_COUNT \
    --temp-dir="$HOME/ray_tmp"

sleep 3
ray status

# ── Start FastAPI ────────────────────────────────────────────────
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 1

echo "$LOG Starting FastAPI..."
cd "$PROJECT_DIR"
nohup uvicorn app.main:app \
    --host 0.0.0.0 --port 8000 --log-level info \
    > "$HOME/kuberay-api.log" 2>&1 &

echo "$!" > "$HOME/kuberay-api.pid"

# ── Wait for healthy ─────────────────────────────────────────────
echo "$LOG Waiting for API to load model..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "$LOG API healthy!"
        break
    fi
    printf "  Loading... (%ds)\r" "$((i * 3))"
    sleep 3
    if [ "$i" -eq 60 ]; then
        echo "$LOG FAILED. Check: tail -30 ~/kuberay-api.log"
        exit 1
    fi
done

# ── Quick test ───────────────────────────────────────────────────
echo ""
echo "$LOG Running test..."
RESPONSE=$(curl -sf -X POST http://localhost:8000/v1/batches \
    -H "Content-Type: application/json" \
    -H "X-API-Key: sk-kuberay-batch-2024" \
    -d '{"input": [{"prompt": "What is 2+2?"}], "max_tokens": 30}')

JOB_ID=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "$LOG Job: $JOB_ID"

for i in $(seq 1 30); do
    RESULT=$(curl -sf "http://localhost:8000/v1/batches/$JOB_ID" -H "X-API-Key: sk-kuberay-batch-2024")
    STATUS=$(echo "$RESULT" | python -c "import sys,json; print(json.load(sys.stdin)['status'])")
    [ "$STATUS" = "completed" ] && break
    [ "$STATUS" = "failed" ] && break
    sleep 2
done

echo "$RESULT" | python -m json.tool
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "$LOG ════════════════════════════════════════"
echo "$LOG  READY"
echo "$LOG  API:       http://localhost:8000"
echo "$LOG  Dashboard: http://localhost:8265"
echo "$LOG  API key:   sk-kuberay-batch-2024"
echo "$LOG  Log:       ~/kuberay-api.log"
echo "$LOG"
echo "$LOG  Stop:  bash scripts/deploy.sh stop"
echo "$LOG ════════════════════════════════════════"
