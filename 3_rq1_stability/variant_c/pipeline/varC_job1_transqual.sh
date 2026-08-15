#!/bin/bash
#SBATCH --job-name=varC_job1_qual
#SBATCH --output=logs/rq1_varC/job1_qual_%j.out
#SBATCH --error=logs/rq1_varC/job1_qual_%j.err
#SBATCH --time=12:00:00
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
echo "RQ1 Variant C — Job 1: Generate varC Transcripts + Qualitative Assessment"
echo "Job started: $(date)"
echo "Node: $HOSTNAME | Job ID: $SLURM_JOB_ID"
echo "========================================"

mkdir -p $HOME/logs/rq1_varC
mkdir -p $AIPSY_ROOT/analysis_output/VariantC/qual
mkdir -p $AIPSY_ROOT/analysis_output/VariantC/quan
mkdir -p $AIPSY_ROOT/analysis_output/VariantC/quan_few_shot
mkdir -p $AIPSY_ROOT/analysis_output/VariantC/meta
mkdir -p $AIPSY_ROOT/rq1_perturbations/variant_c
mkdir -p $HOME/logs/variant_c

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

# --- Activate conda (needed for transcript generation — no GPU required) ---
source ~/.bashrc
conda activate aipsy
echo "Python: $(which python)"

cd $AIPSY_ROOT

# STEP 1: Qualitative Assessment (needs GPU + Ollama)

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: Not enough GPU memory, resubmitting..."
    sbatch $AIPSY_ROOT/slurm/varC_job1_qual.sh
    exit 1
fi

# --- Start Ollama ---
echo "Starting Ollama on $HOSTNAME:11434"
$OLLAMA_BIN serve &
OLLAMA_PID=$!

echo "Waiting for Ollama..."
for i in $(seq 1 30); do
    if $OLLAMA_BIN list > /dev/null 2>&1; then
        echo "Ollama ready!"
        break
    fi
    echo "  Attempt $i/30 - waiting 10s..."
    sleep 10
done

echo "Warming up Gemma 3 27B..."
curl -s -m 1800 -X POST http://localhost:11434/api/chat \
  -d '{"model":"gemma3:27b","messages":[{"role":"user","content":"say hi"}],"stream":false}' \
  > /tmp/warmup_response.json
cat /tmp/warmup_response.json | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('message',{}).get('content','ERROR'))" 2>/dev/null
$OLLAMA_BIN ps

export OLLAMA_NODE=localhost

echo ""
echo "=== Step 1: Qualitative Assessment (38 varC subjects) ==="
python 3_rq1_stability/variant_c/varC_qual_assessment.py
echo "Step 1 done: $(date)"

QUAL_OUT=$AIPSY_ROOT/analysis_output/VariantC/qual/qual_assessment_varC.csv
if [ ! -f "$QUAL_OUT" ]; then
    echo "ERROR: Qual output not found: $QUAL_OUT"
    kill $OLLAMA_PID
    exit 1
fi


N_SUBJECTS=$(tail -n +2 $QUAL_OUT | wc -l)
echo "Qual output confirmed: $QUAL_OUT ($N_SUBJECTS subjects)"
if [ "$N_SUBJECTS" -lt 38 ]; then
    echo "WARNING: Expected 38 subjects in qual output, got $N_SUBJECTS"
    echo "Check for failures in the log above."
fi

echo ""
echo "========================================"
echo "Job 1 complete: $(date)"
echo "Output files:"
echo "  rq1_perturbations/variant_c/   ($N_TRANSCRIPTS varC transcripts)"
echo "  logs/variant_c/                (substitution summary + detail log)"
echo "  analysis_output/VariantC/qual/qual_assessment_varC.csv ($N_SUBJECTS subjects)"
echo "========================================"

kill $OLLAMA_PID
