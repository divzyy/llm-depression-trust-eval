#!/bin/bash
#SBATCH --job-name=varC_test
#SBATCH --output=logs/varC_test_%j.out
#SBATCH --error=logs/varC_test_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
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

mkdir -p $HOME/logs/variant_c
mkdir -p $AIPSY_ROOT/rq1_perturbations/variant_c

source ~/.bashrc
conda activate aipsy
echo "Python: $(which python)"

# --- Run Variant C on 3 test subjects only ---
# Subjects chosen: one clearly depressed (PHQ8>=10), one borderline, one not depressed
# 316 (first in test set), 385, 451
# The script supports --test_ids to override the full 41
cd $AIPSY_ROOT

echo ""
echo "=== Variant C: Frequency Anchor Normalization (TEST RUN: 3 subjects) ==="
python 3_rq1_stability/variant_c/generate_paraphrase_variant_c.py \
    --data_dir $DAIC_ROOT/transcripts \
    --output_dir $AIPSY_ROOT/rq1_perturbations/variant_c \
    --log_dir $HOME/logs/variant_c \
    --test_ids 316 385 451

echo ""
echo "=== Output files ==="
ls -lh $AIPSY_ROOT/rq1_perturbations/variant_c/
echo ""
echo "=== Summary CSV ==="
cat $HOME/logs/variant_c/variant_c_summary_*.csv | tail -20

echo ""
echo "=== Spot-check: diff original vs varC for first subject ==="
echo "--- Original (first 30 Participant lines) ---"
grep "^Participant" $DAIC_ROOT/transcripts/316_TRANSCRIPT.csv | head -30
echo ""
echo "--- Variant C (first 30 Participant lines) ---"
grep "^Participant" $AIPSY_ROOT/rq1_perturbations/variant_c/316_TRANSCRIPT_varC.csv | head -30

echo ""
echo "========================================"
echo "Variant C test run complete: $(date)"
echo "========================================"
