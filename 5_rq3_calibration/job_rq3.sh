#!/bin/bash
#SBATCH --job-name=rq3_run
#SBATCH --output=logs/rq3_%x_%j.out
#SBATCH --error=logs/rq3_%x_%j.err
#SBATCH --time=08:00:00
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

# Usage: sbatch job_rq3.sh <condition>
#   condition: main | no_qual | no_quant | transcript_only | explanation_first
CONDITION=${1:-main}

echo "========================================"
echo "RQ3 run | condition=$CONDITION | $(date)"
echo "Node: $HOSTNAME | Job: $SLURM_JOB_ID"
echo "========================================"

module load CUDA/12.4.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOTDIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OLLAMA_BIN="${OLLAMA_BIN:-$HOME/ollama-018/bin/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/ollama/models}"
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1
export LD_LIBRARY_PATH="${OLLAMA_LIB:-$HOME/ollama-018/lib/ollama/cuda_v13:$HOME/ollama-018/lib/ollama}:/usr/lib64:$LD_LIBRARY_PATH"

unset GPU_DEVICE_ORDINAL ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

mkdir -p "$HOME/logs"
mkdir -p "$ROOTDIR/analysis_output/rq3"

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: Not enough GPU memory, resubmitting..."
    sbatch "$SCRIPT_DIR/job_rq3.sh" "$CONDITION"
    exit 1
fi

echo "Starting Ollama on $HOSTNAME:11434"
$OLLAMA_BIN serve &
OLLAMA_PID=$!

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
  > /tmp/warmup_rq3.json
$OLLAMA_BIN ps

export OLLAMA_NODE=localhost

source ~/.bashrc
conda activate aipsy
echo "Python: $(which python)"

cd "$SCRIPT_DIR"

python 5_rq3_calibration/rq3_run.py --condition "$CONDITION"

echo "========================================"
echo "RQ3 $CONDITION complete: $(date)"
echo "========================================"

kill $OLLAMA_PID
