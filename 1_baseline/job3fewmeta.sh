#!/bin/bash
#SBATCH --job-name=aipsy_fewshot
#SBATCH --output=logs/job3_fewshot_%j.out
#SBATCH --error=logs/job3_fewshot_%j.err
#SBATCH --time=03:30:00
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
mkdir -p $AIPSY_ROOT/analysis_output/quan_gemma_few_shot
mkdir -p $AIPSY_ROOT/analysis_output/qual
mkdir -p $AIPSY_ROOT/agents

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: Not enough GPU memory, resubmitting..."
    sbatch $AIPSY_ROOT/slurm/job3fewmeta.sh
    exit 1
fi

QUAL_OUT=$AIPSY_ROOT/analysis_output/qual/qual_assessment_GEMMA_v2.csv
FEEDBACK_OUT=$AIPSY_ROOT/analysis_output/qual/feedback_assessments_v2.csv
if [ ! -f "$QUAL_OUT" ]; then
    echo "ERROR: Qual assessment output not found at $QUAL_OUT"
    echo "Job 1 must complete successfully before running Job 3."
    exit 1
fi
if [ ! -f "$FEEDBACK_OUT" ]; then
    echo "ERROR: Feedback output not found at $FEEDBACK_OUT"
    echo "Job 1 must complete successfully before running Job 3."
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

echo "Warming up Gemma 3 27B..."
curl -s -m 1800 -X POST http://localhost:11434/api/chat \
  -d '{"model":"gemma3:27b","messages":[{"role":"user","content":"say hi"}],"stream":false}' \
  > /tmp/warmup_response.json
echo "Warmup response:"
cat /tmp/warmup_response.json | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('message',{}).get('content','ERROR'))" 2>/dev/null
echo "GPU status after Gemma warmup:"
$OLLAMA_BIN ps

echo "Pulling Qwen3 Embedding model..."
$OLLAMA_BIN pull qwen3-embedding:8b-q8_0
echo "Embedding pull done: $(date)"

export OLLAMA_NODE=localhost

source ~/.bashrc
conda activate aipsy
echo "Python: $(which python)"

cd $AIPSY_ROOT

echo ""
echo "=== Step 1: Generate embedding pickle (58 training subjects, chunk=8, step=2, dim=4096) ==="
python "$AIPSY_ROOT/1_baseline/generate_pickle.py"
echo "Step 1 done: $(date)"

PICKLE=$AIPSY_ROOT/agents/chunk_8_step_2_participant_embedded_transcripts.pkl
if [ ! -f "$PICKLE" ]; then
    echo "ERROR: Pickle file not found at $PICKLE"
    echo "Cannot proceed with few-shot pipeline."
    kill $OLLAMA_PID
    exit 1
fi
echo "Pickle file confirmed: $PICKLE"

echo ""
echo "=== Step 2: Few-shot Quantitative Assessment (41 test subjects, chunk=8, examples=2, dim=4096) ==="
python 1_baseline/quantitative_assessment/embedding_batch_script.py \
    --ollama_node localhost \
    --chunk_step chunk_8_step_2 \
    --examples_num 2 \
    --dims 4096 \
    --num_runs 1 \
    --eval_split test
echo "Step 2 done: $(date)"

FEWSHOT=$AIPSY_ROOT/analysis_output/quan_gemma_few_shot/ids_test_chunk_8_step_2_dim_4096_examples_2_embedding_results_analysis_1.jsonl
if [ ! -f "$FEWSHOT" ]; then
    echo "ERROR: Few-shot output not found at $FEWSHOT"
    echo "Cannot proceed with meta-review."
    kill $OLLAMA_PID
    exit 1
fi
echo "Few-shot output confirmed: $FEWSHOT"

echo ""
echo "=== Step 3: Meta Review (few-shot quantitative input) ==="
python 1_baseline/meta_review/meta_review.py
echo "Step 3 done: $(date)"

META_OUT=$AIPSY_ROOT/analysis_output/qual/meta_review_fewshot_test_v2.csv
if [ ! -f "$META_OUT" ]; then
    echo "ERROR: Meta-review (few-shot) output not found at $META_OUT"
    kill $OLLAMA_PID
    exit 1
fi
echo "Meta-review (few-shot) output confirmed: $META_OUT"

echo ""
echo "========================================"
echo "Job 3 complete: $(date)"
echo "Output files:"
echo "  agents/chunk_8_step_2_dim_4096_participant_embedded_transcripts.pkl"
echo "  agents/split_manifest_58_43_41.json"
echo "  analysis_output/quan_gemma_few_shot/ids_test_chunk_8_step_2_dim_4096_examples_2_embedding_results_analysis_1.jsonl"
echo "  analysis_output/qual/meta_review_fewshot_test_v2.csv"
echo "========================================"

kill $OLLAMA_PID
