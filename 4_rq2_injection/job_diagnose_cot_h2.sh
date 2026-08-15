#!/bin/bash
#SBATCH --job-name=cot_h2_diag
#SBATCH --output=logs/cot_h2_diag_%j.out
#SBATCH --error=logs/cot_h2_diag_%j.err
#SBATCH --time=01:00:00
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


echo "Job started: $(date) on $HOSTNAME (id $SLURM_JOB_ID)"

module load CUDA/12.4.0

OLLAMA_BIN=$OLLAMA_HOME/bin/ollama
export OLLAMA_MODELS=$HOME/ollama/models
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1
export LD_LIBRARY_PATH=$OLLAMA_HOME/lib/ollama/cuda_v13:$OLLAMA_HOME/lib/ollama:/usr/lib64:$LD_LIBRARY_PATH

unset GPU_DEVICE_ORDINAL ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
mkdir -p $HOME/logs

for attempt in $(seq 1 10); do
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "Free GPU memory: ${FREE_MEM} MiB (attempt $attempt/10)"
    [ "$FREE_MEM" -ge "17000" ] && break
    echo "  waiting 60s for the GPU..."
    sleep 60
done
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: GPU still busy after 10 minutes. Exiting; resubmit later."
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
  > /tmp/warmup_cot_h2.json
echo "Warmup:"; cat /tmp/warmup_cot_h2.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',{}).get('content','ERROR'))" 2>/dev/null
$OLLAMA_BIN ps

export OLLAMA_NODE=localhost
source ~/.bashrc
conda activate aipsy

cd $AIPSY_ROOT
echo ""
echo "=== Running the CoT / H2 diagnostic ==="
python scratch/diagnose_cot_h2.py

kill $OLLAMA_PID 2>/dev/null || true
echo ""
echo "Job complete: $(date)"
echo "Read: rq2_proinject/cot_h2_audit_text.txt"
