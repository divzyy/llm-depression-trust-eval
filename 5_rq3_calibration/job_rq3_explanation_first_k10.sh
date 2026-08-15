#!/bin/bash
#SBATCH --job-name=rq3_expfirst_k10
#SBATCH --output=logs/rq3_expfirst_%j.out
#SBATCH --error=logs/rq3_expfirst_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=gpu-short

AIPSY_ROOT="${AIPSY_ROOT:-$HOME/ai-psychiatrist}"
DAIC_ROOT="${DAIC_ROOT:-$HOME/daic_woz_data}"
OLLAMA_HOME="${OLLAMA_HOME:-$HOME/ollama-018}"
export AIPSY_ROOT DAIC_ROOT

# RQ3 — reasoning-first (explanation_first) rerun at K=10 with S2 enabled.

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module load CUDA/12.4.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOTDIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OLLAMA_BIN="${OLLAMA_BIN:-$HOME/ollama-018/bin/ollama}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS:-$HOME/ollama/models}"

export OLLAMA_MODELS=$OLLAMA_MODELS_DIR
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_DEBUG=1
export LD_LIBRARY_PATH="${OLLAMA_LIB:-$HOME/ollama-018/lib/ollama/cuda_v13:$HOME/ollama-018/lib/ollama}:/usr/lib64:$LD_LIBRARY_PATH"

unset GPU_DEVICE_ORDINAL
unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

mkdir -p "$HOME/logs"
mkdir -p "$ROOTDIR/analysis_output/rq3"

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: Not enough GPU memory. Exiting."
    exit 1
fi

echo "Starting Ollama on $HOSTNAME:11434"
$OLLAMA_BIN serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
for i in $(seq 1 30); do
    if $OLLAMA_BIN list > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    echo "  Attempt $i/30 - waiting 10s..."
    sleep 10
done

echo "Warming up Gemma 3 27B..."
curl -s -m 1800 -X POST http://localhost:11434/api/chat \
  -d '{"model":"gemma3:27b","messages":[{"role":"user","content":"say hi"}],"stream":false}' \
  > /tmp/warmup_rq3_expfirst.json
echo "Warmup response:"
cat /tmp/warmup_rq3_expfirst.json | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('message',{}).get('content','ERROR'))" 2>/dev/null
echo "GPU status after warmup:"
$OLLAMA_BIN ps


export OLLAMA_NODE=localhost

source ~/.bashrc
conda activate aipsy

# --- Run RQ3 explanation_first at K=10 ---
cd "$ROOTDIR"
echo ""
if [ -n "${SUBJECTS:-}" ]; then
    echo "=== RQ3 explanation_first | K=10 | SMOKE TEST subjects: $SUBJECTS ==="
    python 5_rq3_calibration/rq3_run.py --condition explanation_first --k 10 --subjects "$SUBJECTS"
else
    echo "=== RQ3 explanation_first | K=10 | all 41 subjects ==="
    python 5_rq3_calibration/rq3_run.py --condition explanation_first --k 10
fi
echo "RQ3 explanation_first done: $(date)"

kill $OLLAMA_PID

echo ""
echo "========================================"
echo "Job complete: $(date)"
echo "Result: analysis_output/rq3/rq3_explanation_first_meta.csv"
echo "========================================"
