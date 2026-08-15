#!/bin/bash
#SBATCH --job-name=aipsy_qual
#SBATCH --output=logs/job1_qual_%j.out
#SBATCH --error=logs/job1_qual_%j.err
#SBATCH --time=24:00:00
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
mkdir -p $AIPSY_ROOT/analysis_output/qual

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: Not enough GPU memory, resubmitting..."
    sbatch $AIPSY_ROOT/slurm/job1qualfeedback.sh
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
  > /tmp/warmup_response.json
echo "Warmup response:"
cat /tmp/warmup_response.json | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('message',{}).get('content','ERROR'))" 2>/dev/null
echo "GPU status after warmup:"
$OLLAMA_BIN ps

export OLLAMA_NODE=localhost

source ~/.bashrc
conda activate aipsy
echo "Python: $(which python)"

cd $AIPSY_ROOT

echo ""
echo "=== Step 1: Qualitative Assessment (142 subjects) ==="
python 1_baseline/qualitative_assessment/qual_assessment.py
echo "Step 1 done: $(date)"

echo ""
echo "=== Step 2: Judge + Feedback Loop ==="
python 1_baseline/qualitative_assessment/feedback_loop.py
echo "Step 2 done: $(date)"

QUAL_OUT=$AIPSY_ROOT/analysis_output/qual/qual_assessment_GEMMA_v2.csv
FEEDBACK_OUT=$AIPSY_ROOT/analysis_output/qual/feedback_assessments_v2.csv
if [ ! -f "$QUAL_OUT" ]; then
    echo "ERROR: Qual assessment output not found at $QUAL_OUT"
    kill $OLLAMA_PID
    exit 1
fi
if [ ! -f "$FEEDBACK_OUT" ]; then
    echo "ERROR: Feedback output not found at $FEEDBACK_OUT"
    kill $OLLAMA_PID
    exit 1
fi
echo "Qual output confirmed: $QUAL_OUT"
echo "Feedback output confirmed: $FEEDBACK_OUT"

echo ""
echo "========================================"
echo "Job 1 complete: $(date)"
echo "Output files:"
echo "  analysis_output/qual/qual_assessment_GEMMA_v2.csv"
echo "  analysis_output/qual/feedback_assessments_v2.csv"
echo "  analysis_output/qual/feedback_evaluations_v2.csv"
echo "========================================"

kill $OLLAMA_PID
