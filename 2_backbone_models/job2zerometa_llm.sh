#!/bin/bash
#SBATCH --job-name=aipsy_zeroshot
#SBATCH --output=logs/llm_job2_zeroshot_%j.out
#SBATCH --error=logs/llm_job2_zeroshot_%j.err
#SBATCH --time=4:00:00
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

# Options: gemma3:27b | medgemma:27b | llama3.1:8b | deepseek-r1:14b
MODEL_NAME="${MODEL_NAME:-alibayram/medgemma:27b}"

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL_NAME"
echo "========================================"

module load CUDA/12.4.0

OLLAMA_BIN=$OLLAMA_HOME/bin/ollama
OLLAMA_MODELS_DIR=$HOME/ollama/models

export OLLAMA_MODELS=$OLLAMA_MODELS_DIR
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_DEBUG=0
export LD_LIBRARY_PATH=$OLLAMA_HOME/lib/ollama/cuda_v13:$OLLAMA_HOME/lib/ollama:/usr/lib64:$LD_LIBRARY_PATH

export MODEL_NAME=$MODEL_NAME

unset GPU_DEVICE_ORDINAL
unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES


MODEL_TAG=$(echo "$MODEL_NAME" | tr ':/' '__')

MODEL_DISPLAY=$(echo "$MODEL_NAME" | sed 's|.*/||' | cut -d: -f1 | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')

mkdir -p $HOME/logs
mkdir -p "$AIPSY_ROOT/analysis_output/LLMs Study/${MODEL_DISPLAY}/quan"
mkdir -p "$AIPSY_ROOT/analysis_output/LLMs Study/${MODEL_DISPLAY}/qual"

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: Not enough GPU memory, resubmitting..."
    sbatch "$AIPSY_ROOT/LLMs Study/job2zerometa_llm.sh"
    exit 1
fi

QUAL_OUT="$AIPSY_ROOT/analysis_output/LLMs Study/${MODEL_DISPLAY}/qual/qual_assessment_${MODEL_TAG}_v2.csv"
if [ ! -f "$QUAL_OUT" ]; then
    echo "ERROR: Qual assessment output not found at $QUAL_OUT"
    echo "Job 1 must complete successfully before running Job 2."
    exit 1
fi
echo "Job 1 outputs confirmed, proceeding."

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

echo "Pulling model: $MODEL_NAME ..."
$OLLAMA_BIN pull "$MODEL_NAME"
echo "Pull done: $(date)"

echo "Warming up model: $MODEL_NAME ..."
curl -s -m 1800 -X POST http://localhost:11434/api/chat \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}],\"stream\":false}" \
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
echo "=== Step 1: Zero-shot Quantitative Assessment (41 test subjects) ==="
python "LLMs Study/quantitative_analysis_llm.py"
echo "Step 1 done: $(date)"

ZEROSHOT="$AIPSY_ROOT/analysis_output/LLMs Study/${MODEL_DISPLAY}/quan/results_zero_shot_test41_detailed.jsonl"
if [ ! -f "$ZEROSHOT" ]; then
    echo "ERROR: Zero-shot output not found at $ZEROSHOT"
    echo "Cannot proceed with meta-review."
    kill $OLLAMA_PID
    exit 1
fi
echo "Zero-shot output confirmed: $ZEROSHOT"

echo ""
echo "=== Step 2: Meta Review (zero-shot quantitative input) ==="
python "LLMs Study/meta_review_zeroshot_llm.py"
echo "Step 2 done: $(date)"

META_OUT="$AIPSY_ROOT/analysis_output/LLMs Study/${MODEL_DISPLAY}/qual/meta_review_zeroshot_test_v2.csv"
if [ ! -f "$META_OUT" ]; then
    echo "ERROR: Meta-review (zero-shot) output not found at $META_OUT"
    kill $OLLAMA_PID
    exit 1
fi
echo "Meta-review (zero-shot) output confirmed: $META_OUT"

echo ""
echo "========================================"
echo "Job 2 complete: $(date)"
echo "Model: $MODEL_NAME"
echo "Output files:"
echo "  analysis_output/LLMs Study/${MODEL_DISPLAY}/quan/results_zero_shot_test41.csv"
echo "  analysis_output/LLMs Study/${MODEL_DISPLAY}/quan/results_zero_shot_test41_detailed.jsonl"
echo "  analysis_output/LLMs Study/${MODEL_DISPLAY}/qual/meta_review_zeroshot_test_v2.csv"
echo "========================================"

kill $OLLAMA_PID
