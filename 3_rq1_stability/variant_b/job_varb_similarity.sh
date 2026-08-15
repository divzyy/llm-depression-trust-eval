#!/bin/bash
#SBATCH --job-name=varb_similarity
#SBATCH --output=logs/varb_similarity_%j.out
#SBATCH --error=logs/varb_similarity_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-short

AIPSY_ROOT="${AIPSY_ROOT:-$HOME/ai-psychiatrist}"
DAIC_ROOT="${DAIC_ROOT:-$HOME/daic_woz_data}"
OLLAMA_HOME="${OLLAMA_HOME:-$HOME/ollama-018}"
export AIPSY_ROOT DAIC_ROOT

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source ~/.bashrc
conda activate aipsy
echo "Python: $(which python)"

# --- Run semantic similarity ---
cd $AIPSY_ROOT
echo ""
echo "=== Variant B: Semantic Similarity (all 41 subjects) ==="
python 3_rq1_stability/variant_b/semantic_similarity.py \
    --orig_dir $DAIC_ROOT/transcripts \
    --varb_dir $AIPSY_ROOT/rq1_perturbations/variant_b \
    --log_dir  $HOME/logs/variant_b
echo "Similarity done: $(date)"

echo ""
echo "========================================"
echo "Log dir: $HOME/logs/variant_b/"
echo "========================================"
