#!/bin/bash
#SBATCH --job-name=rq3_reduced_k10
#SBATCH --output=logs/rq3_reduced_k10_%j.out
#SBATCH --error=logs/rq3_reduced_k10_%j.err
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

# Re-run the three remaining reduced conditions at K=10

echo "Job started: $(date) on $HOSTNAME (id $SLURM_JOB_ID)"

module load CUDA/12.4.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOTDIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OLLAMA_BIN="${OLLAMA_BIN:-$HOME/ollama-018/bin/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/ollama/models}"
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_DEBUG=1
export LD_LIBRARY_PATH="${OLLAMA_LIB:-$HOME/ollama-018/lib/ollama/cuda_v13:$HOME/ollama-018/lib/ollama}:/usr/lib64:$LD_LIBRARY_PATH"

unset GPU_DEVICE_ORDINAL ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
mkdir -p "$HOME/logs"
mkdir -p "$ROOTDIR/analysis_output/rq3"

for attempt in $(seq 1 10); do
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "Free GPU memory: ${FREE_MEM} MiB (attempt $attempt/10)"
    [ "$FREE_MEM" -ge "17000" ] && break
    echo "  waiting 60s for the GPU..."
    sleep 60
done
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: GPU still busy after 10 minutes. Exiting; the chained job will retry."
    exit 1
fi

echo "Starting Ollama on $HOSTNAME:11434"
$OLLAMA_BIN serve &
OLLAMA_PID=$!

for i in $(seq 1 30); do
    if $OLLAMA_BIN list > /dev/null 2>&1; then echo "Ollama is ready!"; break; fi
    echo "  Attempt $i/30 - waiting 10s..."; sleep 10
done

curl -s -m 1800 -X POST http://localhost:11434/api/chat \
  -d '{"model":"gemma3:27b","messages":[{"role":"user","content":"say hi"}],"stream":false}' \
  > /tmp/warmup_rq3_reduced.json
echo "Warmup:"; cat /tmp/warmup_rq3_reduced.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',{}).get('content','ERROR'))" 2>/dev/null
$OLLAMA_BIN ps

export OLLAMA_NODE=localhost
source ~/.bashrc
conda activate aipsy

cd "$ROOTDIR"

for COND in no_qual no_quant transcript_only; do
    echo ""
    echo "############################################################"
    echo "### condition: $COND at K=10   ($(date))"
    echo "############################################################"
    python 5_rq3_calibration/rq3_run.py --condition "$COND" --k 10
    echo "### $COND finished or interrupted at $(date)"
done

kill $OLLAMA_PID 2>/dev/null || true
echo ""
echo "Job complete: $(date)"
echo "Files: analysis_output/rq3/rq3_{no_qual,no_quant,transcript_only}_meta.csv"
