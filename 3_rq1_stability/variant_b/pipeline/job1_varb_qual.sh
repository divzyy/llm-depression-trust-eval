#!/bin/bash
#SBATCH --job-name=varB_qual_new
#SBATCH --output=logs/rq1_varB_new/job1_qual_%j.out
#SBATCH --error=logs/rq1_varB_new/job1_qual_%j.err
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
echo "RQ1 Variant B — Job 1: Qualitative Assessment (41 subjects)"
echo "Job started: $(date)"
echo "Node: $HOSTNAME | Job ID: $SLURM_JOB_ID"
echo "========================================"

mkdir -p $HOME/logs/rq1_varB_new
mkdir -p $AIPSY_ROOT/analysis_output/VariantB/new/qual
mkdir -p $AIPSY_ROOT/analysis_output/VariantB/new/quan_zero_shot
mkdir -p $AIPSY_ROOT/analysis_output/VariantB/new/quan_few_shot
mkdir -p $AIPSY_ROOT/analysis_output/VariantB/new/meta

module load CUDA/12.4.0
OLLAMA_BIN=$OLLAMA_HOME/bin/ollama
OLLAMA_MODELS_DIR=$HOME/ollama/models
export OLLAMA_MODELS=$OLLAMA_MODELS_DIR
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_DEBUG=1
export LD_LIBRARY_PATH=$OLLAMA_HOME/lib/ollama/cuda_v13:$OLLAMA_HOME/lib/ollama:/usr/lib64:$LD_LIBRARY_PATH
unset GPU_DEVICE_ORDINAL ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: Not enough GPU memory, resubmitting..."
    sbatch $AIPSY_ROOT/slurm/job1_varb_qual.sh
    exit 1
fi

echo "Cleaning up any existing Ollama processes..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 5
if ss -tlnp 2>/dev/null | grep -q ":11434"; then
    echo "Port 11434 still in use, force-killing..."
    fuser -k 11434/tcp 2>/dev/null || true
    sleep 3
fi

$OLLAMA_BIN serve &
OLLAMA_PID=$!
for i in $(seq 1 30); do
    if $OLLAMA_BIN list > /dev/null 2>&1; then echo "Ollama ready!"; break; fi
    echo "  Attempt $i/30..."; sleep 10
done

echo "Warming up Gemma 3 27B..."
curl -s -m 1800 -X POST http://localhost:11434/api/chat \
  -d '{"model":"gemma3:27b","messages":[{"role":"user","content":"say hi"}],"stream":false}' \
  > /tmp/warmup_response.json
$OLLAMA_BIN ps

export OLLAMA_NODE=localhost
source ~/.bashrc
conda activate aipsy
cd $AIPSY_ROOT

echo ""
echo "=== Qualitative Assessment — RQ1 Variant B (41 subjects) ==="
python 3_rq1_stability/variant_b/varb_qual_assessment.py
echo "Qual done: $(date)"

QUAL_OUT=$AIPSY_ROOT/analysis_output/VariantB/new/qual/qual_assessment_GEMMA_v2.csv
if [ ! -f "$QUAL_OUT" ]; then
    echo "ERROR: Qual output not found: $QUAL_OUT"
    kill $OLLAMA_PID
    exit 1
fi
N_SUBJECTS=$(tail -n +2 $QUAL_OUT | wc -l)
echo "Qual output confirmed: $QUAL_OUT ($N_SUBJECTS subjects)"

echo ""
echo "========================================"
echo "Job 1 complete: $(date)"
echo "Output: analysis_output/VariantB/new/qual/qual_assessment_GEMMA_v2.csv"
echo "========================================"

kill $OLLAMA_PID
