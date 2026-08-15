#!/bin/bash
#SBATCH --job-name=variantB_paraphrase_full
#SBATCH --output=logs/variantB_full_%j.out
#SBATCH --error=logs/variantB_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=gpu-mig-40g

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

mkdir -p $HOME/logs/variant_b/new
mkdir -p $AIPSY_ROOT/rq1_perturbations/variant_b/new

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: Not enough GPU memory, resubmitting..."
    sbatch $AIPSY_ROOT/job_variant_b.sh
    exit 1
fi

# Kill any existing Ollama to avoid "address already in use" error
echo "Cleaning up any existing Ollama processes..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 5
if ss -tlnp 2>/dev/null | grep -q ":11434"; then
    echo "Port 11434 still in use, force-killing..."
    fuser -k 11434/tcp 2>/dev/null || true
    sleep 3
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
  > /tmp/warmup_response.json
echo "Warmup response:"
cat /tmp/warmup_response.json | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('message',{}).get('content','ERROR'))" 2>/dev/null
echo "GPU status after warmup:"
$OLLAMA_BIN ps

# --- Set OLLAMA_NODE for Python script ---
export OLLAMA_NODE=localhost

source ~/.bashrc
conda activate aipsy
echo "Python: $(which python)"

# --- Run Variant B paraphrase generation ---
cd $AIPSY_ROOT
echo ""
echo "=== Variant B: Paraphrase generation (full run: all 41 subjects) ==="
python 3_rq1_stability/variant_b/generate_paraphrase_variant_b.py \
    --data_dir   $DAIC_ROOT/transcripts \
    --output_dir $AIPSY_ROOT/rq1_perturbations/variant_b/new \
    --log_dir    $HOME/logs/variant_b/new
echo "Variant B done: $(date)"

echo ""
echo "========================================"
echo "Output directory: $AIPSY_ROOT/rq1_perturbations/variant_b/new"
echo "Log directory:    $HOME/logs/variant_b/new"
echo "========================================"

kill $OLLAMA_PID
