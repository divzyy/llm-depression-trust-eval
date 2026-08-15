#!/bin/bash
# Submit full LLM Study pipeline for 3 models: DeepSeek, LLaMA, MedGemma
# Each model: Job1 (qual+feedback) -> Job2 (zero-shot+meta) + Job3 (few-shot+meta)

set -e
cd $AIPSY_ROOT

# --- Step 1: Clean existing outputs ---
echo "=== Cleaning existing outputs ==="
rm -rf "$AIPSY_ROOT/analysis_output/LLMs Study/Deepseek-r1"
rm -rf "$AIPSY_ROOT/analysis_output/LLMs Study/Llama3.1"
rm -rf "$AIPSY_ROOT/analysis_output/LLMs Study/Medgemma"
echo "Done cleaning."

# --- Step 2: Submit jobs ---
echo ""
echo "=== Submitting DeepSeek-R1:14b ==="
DS_J1=$(sbatch --export=ALL,MODEL_NAME="deepseek-r1:14b" --parsable "LLMs Study/job1qualfeedback_llm.sh")
echo "  Job1: $DS_J1"
DS_J2=$(sbatch --export=ALL,MODEL_NAME="deepseek-r1:14b" --dependency=afterok:$DS_J1 --parsable "LLMs Study/job2zerometa_llm.sh")
echo "  Job2: $DS_J2 (depends on $DS_J1)"
DS_J3=$(sbatch --export=ALL,MODEL_NAME="deepseek-r1:14b" --dependency=afterok:$DS_J1 --parsable "LLMs Study/job3fewmeta_llm.sh")
echo "  Job3: $DS_J3 (depends on $DS_J1)"

echo ""
echo "=== Submitting LLaMA 3.1:8b ==="
LL_J1=$(sbatch --export=ALL,MODEL_NAME="llama3.1:8b" --parsable "LLMs Study/job1qualfeedback_llm.sh")
echo "  Job1: $LL_J1"
LL_J2=$(sbatch --export=ALL,MODEL_NAME="llama3.1:8b" --dependency=afterok:$LL_J1 --parsable "LLMs Study/job2zerometa_llm.sh")
echo "  Job2: $LL_J2 (depends on $LL_J1)"
LL_J3=$(sbatch --export=ALL,MODEL_NAME="llama3.1:8b" --dependency=afterok:$LL_J1 --parsable "LLMs Study/job3fewmeta_llm.sh")
echo "  Job3: $LL_J3 (depends on $LL_J1)"

echo ""
echo "=== Submitting MedGemma:27b ==="
MG_J1=$(sbatch --export=ALL,MODEL_NAME="alibayram/medgemma:27b" --parsable "LLMs Study/job1qualfeedback_llm.sh")
echo "  Job1: $MG_J1"
MG_J2=$(sbatch --export=ALL,MODEL_NAME="alibayram/medgemma:27b" --dependency=afterok:$MG_J1 --parsable "LLMs Study/job2zerometa_llm.sh")
echo "  Job2: $MG_J2 (depends on $MG_J1)"
MG_J3=$(sbatch --export=ALL,MODEL_NAME="alibayram/medgemma:27b" --dependency=afterok:$MG_J1 --parsable "LLMs Study/job3fewmeta_llm.sh")
echo "  Job3: $MG_J3 (depends on $MG_J1)"

echo ""
echo "=== Summary ==="
echo "DeepSeek:  J1=$DS_J1  J2=$DS_J2  J3=$DS_J3"
echo "LLaMA:     J1=$LL_J1  J2=$LL_J2  J3=$LL_J3"
echo "MedGemma:  J1=$MG_J1  J2=$MG_J2  J3=$MG_J3"
echo ""
echo "=== Current Queue ==="
squeue -u "$USER"
