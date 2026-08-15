#!/bin/bash
#SBATCH --job-name=varD_fewshot
#SBATCH --output=logs/rq1_varD/job3_fewshot_%j.out
#SBATCH --error=logs/rq1_varD/job3_fewshot_%j.err
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

# ── (rate, seed) for this job. Default rate_10 / seed_1. Override: sbatch job3_vard_fewshot.sh RATE SEED ──
RATE="${1:-10}"
SEED="${2:-1}"
THIS_JOB=$AIPSY_ROOT/3_rq1_stability/variant_d/pipeline/job3_vard_fewshot.sh  

echo "========================================"
echo "RQ1 Variant D — Job 3: Few-shot Quan + Meta (41 subjects)"
echo "Rate: ${RATE}%  Seed: ${SEED}"
echo "Job started: $(date)"
echo "Node: $HOSTNAME | Job ID: $SLURM_JOB_ID"
echo "========================================"

mkdir -p $HOME/logs/rq1_varD
mkdir -p $AIPSY_ROOT/analysis_output/VariantD/rate_${RATE}/seed_${SEED}/quan_few_shot
mkdir -p $AIPSY_ROOT/analysis_output/VariantD/rate_${RATE}/seed_${SEED}/meta

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

QUAL_OUT=$AIPSY_ROOT/analysis_output/VariantD/rate_${RATE}/seed_${SEED}/qual/qual_assessment_GEMMA_v2.csv
if [ ! -f "$QUAL_OUT" ]; then
    echo "ERROR: Job 1 qual output not found for rate ${RATE} seed ${SEED}. Run job1_vard_qual first."
    exit 1
fi
echo "Job 1 output confirmed: $QUAL_OUT"

PICKLE=$AIPSY_ROOT/agents/chunk_8_step_2_participant_embedded_transcripts.pkl
if [ ! -f "$PICKLE" ]; then
    echo "ERROR: Pickle not found: $PICKLE"
    exit 1
fi
echo "Pickle confirmed (reusing baseline): $PICKLE"

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "$FREE_MEM" -lt "17000" ]; then
    echo "ERROR: Not enough GPU memory, resubmitting..."
    sbatch $THIS_JOB $RATE $SEED
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

echo "Pulling Qwen3 Embedding model (before Gemma loads into GPU)..."
$OLLAMA_BIN pull qwen3-embedding:8b-q8_0 || echo "WARNING: ollama pull failed — using cached model"

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
echo "=== Step 1: Few-shot Quantitative Assessment — Variant D (rate ${RATE}%, seed ${SEED}) ==="
python 3_rq1_stability/variant_d/pipeline/quan_few_shot_vard.py \
    --ollama_node localhost \
    --chunk_step chunk_8_step_2 \
    --examples_num 2 \
    --dims 4096 \
    --num_runs 1 \
    --eval_split test \
    --rate $RATE \
    --seed $SEED
echo "Step 1 done: $(date)"

FEWSHOT=$AIPSY_ROOT/analysis_output/VariantD/rate_${RATE}/seed_${SEED}/quan_few_shot/ids_test_chunk_8_step_2_dim_4096_examples_2_embedding_results_analysis_1.jsonl
if [ ! -f "$FEWSHOT" ]; then
    echo "ERROR: Few-shot output not found: $FEWSHOT"
    kill $OLLAMA_PID
    exit 1
fi
echo "Few-shot output confirmed: $FEWSHOT"

echo ""
echo "=== Step 2: Meta Review — Few-shot (Variant D) ==="
python 3_rq1_stability/variant_d/pipeline/meta_fewshot_vard.py --rate $RATE --seed $SEED
echo "Step 2 done: $(date)"

META_OUT=$AIPSY_ROOT/analysis_output/VariantD/rate_${RATE}/seed_${SEED}/meta/meta_review_fewshot_test_v2.csv
if [ ! -f "$META_OUT" ]; then
    echo "ERROR: Meta-review output not found: $META_OUT"
    kill $OLLAMA_PID
    exit 1
fi
echo "Meta-review confirmed: $META_OUT"

echo ""
echo "========================================"
echo "Job 3 complete: $(date)"
echo "Output files:"
echo "  analysis_output/VariantD/rate_${RATE}/seed_${SEED}/quan_few_shot/ids_test_chunk_8_step_2_dim_4096_examples_2_embedding_results_analysis_1.jsonl"
echo "  analysis_output/VariantD/rate_${RATE}/seed_${SEED}/meta/meta_review_fewshot_test_v2.csv"
echo "========================================"

kill $OLLAMA_PID
