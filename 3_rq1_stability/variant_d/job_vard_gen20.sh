#!/bin/bash
#SBATCH --job-name=vard_gen_20
#SBATCH --output=logs/vard_gen_20_%j.out
#SBATCH --error=logs/vard_gen_20_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=gpu-mig-40g

AIPSY_ROOT="${AIPSY_ROOT:-$HOME/ai-psychiatrist}"
DAIC_ROOT="${DAIC_ROOT:-$HOME/daic_woz_data}"
OLLAMA_HOME="${OLLAMA_HOME:-$HOME/ollama-018}"
export AIPSY_ROOT DAIC_ROOT

echo "========================================"
echo "Variant D — Rate 20%"
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
mkdir -p $AIPSY_ROOT/rq1_perturbations/variant_d/rate_20

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: Not enough GPU memory"
    exit 1
fi

# --- Start Ollama ---
echo "Starting Ollama on $HOSTNAME:11434"
$OLLAMA_BIN serve &
OLLAMA_PID=$!

echo "Waiting for Ollama..."
for i in $(seq 1 30); do
    if $OLLAMA_BIN list > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    echo "  Attempt $i/30 - waiting 10s..."
    sleep 10
done

echo "Warming up Gemma 3 27B..."
curl -s -m 1800 -X POST http://localhost:11434/api/chat   -d '{"model":"gemma3:27b","messages":[{"role":"user","content":"say hi"}],"stream":false}'   > /tmp/warmup_20.json
echo "Warmup done. GPU status:"
$OLLAMA_BIN ps

source ~/.bashrc
conda activate aipsy
echo "Python: $(which python)"

# --- Run generation for rate 20% ---
export OLLAMA_NODE=localhost
cd $AIPSY_ROOT

echo ""
echo "=== Generating transcripts: rate=20%, seeds=1-5 ==="
python 3_rq1_stability/variant_d/generate_variant_d.py --rate 20
echo "Generation done: $(date)"

echo ""
echo "Checking transcript counts:"
for seed in 1 2 3 4 5; do
    DIR=$AIPSY_ROOT/rq1_perturbations/variant_d/rate_20/seed_${seed}
    COUNT=$(ls $DIR/*.csv 2>/dev/null | wc -l)
    echo "  rate_20/seed_${seed}: ${COUNT}/41 transcripts"
done

LOG=$AIPSY_ROOT/rq1_perturbations/variant_d/rate_20/generation_log.csv
if [ -f "$LOG" ]; then
    ROWS=$(tail -n +2 $LOG | wc -l)
    echo "  generation_log.csv: OK (${ROWS} entries, expected 205)"
else
    echo "  WARNING: generation_log.csv not found"
fi

echo ""
echo "========================================"
echo "Job complete: $(date)"
echo "Output: rq1_perturbations/variant_d/rate_20/"
echo "========================================"

kill $OLLAMA_PID
