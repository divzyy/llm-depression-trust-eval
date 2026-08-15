#!/bin/bash
#SBATCH --job-name=rq2_heldout
#SBATCH --output=logs/rq2_heldout_%j.out
#SBATCH --error=logs/rq2_heldout_%j.err
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


echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module load CUDA/12.4.0

OLLAMA_BIN=$OLLAMA_HOME/bin/ollama
OLLAMA_MODELS_DIR=$HOME/ollama/models

export OLLAMA_MODELS=$OLLAMA_MODELS_DIR
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_DEBUG=1
export LD_LIBRARY_PATH=$OLLAMA_HOME/lib/ollama/cuda_v13:$OLLAMA_HOME/lib/ollama:/usr/lib64:$LD_LIBRARY_PATH

unset GPU_DEVICE_ORDINAL
unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

mkdir -p $HOME/logs
mkdir -p $AIPSY_ROOT/analysis_output

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
  > /tmp/warmup_rq2_heldout.json
echo "Warmup response:"
cat /tmp/warmup_rq2_heldout.json | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('message',{}).get('content','ERROR'))" 2>/dev/null
echo "GPU status after warmup:"
$OLLAMA_BIN ps

export OLLAMA_NODE=localhost

source ~/.bashrc
conda activate aipsy

# --- Run held-out generalisation experiment ---
cd $AIPSY_ROOT
echo ""
if [ -n "${SUBJECTS:-}" ]; then
    echo "=== RQ2 held-out | chunk: $SUBJECTS ==="
else
    echo "=== RQ2 held-out | all remaining participants ==="
fi
python 4_rq2_injection/run_heldout_vpi_experiment.py
echo "Held-out experiment done: $(date)"

kill $OLLAMA_PID

echo ""
echo "========================================"
echo "Job complete: $(date)"
echo "Raw:     analysis_output/... -> rq2_proinject/vpi_heldout_results.csv"
echo "Summary: rq2_proinject/vpi_heldout_summary.csv"
echo "If participants remain, just resubmit - it resumes automatically."
echo "========================================"
