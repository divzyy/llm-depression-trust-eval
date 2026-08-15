# Trust Evaluation for LLM-based Depression Assessment

Code for evaluating an LLM pipeline that scores the PHQ-8 from clinical interviews, by changing only what the pipeline reads and recording what it returns. The pipeline itself is the [AI Psychiatrist Assistant](https://github.com/trendscenter/ai-psychiatrist) and its internal logic is not modified.

## Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Create the conda environment from the provided [`environment.yml`](environment.yml) file:
```bash
cd llm-depression-trust-eval
conda env create --name aipsy --file ./environment.yml
```

3. Activate the environment:
```bash
conda activate aipsy
```

4. Set the paths. The job scripts export these and the Python scripts read them:
```bash
export AIPSY_ROOT=/path/to/this/repository
export DAIC_ROOT=/path/to/daic_woz_data
export OLLAMA_HOME=$HOME/ollama-018
export OLLAMA_MODELS=$HOME/ollama/models
```

5. Create the log directory:
```bash
mkdir -p $AIPSY_ROOT/logs
```

The DAIC-WOZ corpus is not included. Request it from [USC ICT](https://dcapswoz.ict.usc.edu/) and place it at `$DAIC_ROOT` with `labels/` and `transcripts/` subdirectories.

## Ollama

Each job starts its own Ollama server on the compute node, so no separate server job is needed. Pull the models once before submitting anything:

```bash
$OLLAMA_HOME/bin/ollama pull gemma3:27b
$OLLAMA_HOME/bin/ollama pull qwen3-embedding:8b-q8_0
$OLLAMA_HOME/bin/ollama pull alibayram/medgemma:27b
$OLLAMA_HOME/bin/ollama pull llama3.1:8b
$OLLAMA_HOME/bin/ollama pull deepseek-r1:14b
```

The job scripts set `OLLAMA_NODE` themselves from the node they land on. The runs used Ollama 0.18.0.

## Running the Experiments

Submit every job from the repository root. The baseline runs first, because the later stages compare against the reference run it produces.

1. Baseline. Run the three jobs in order:
```bash
sbatch 1_baseline/job1qualfeedback.sh
sbatch 1_baseline/job2zerometa.sh
sbatch 1_baseline/job3fewmeta.sh
```

2. Backbone models. This submits a three-job chain per model with `--dependency=afterok`:
```bash
bash 2_backbone_models/submit_llm_study.sh
```
A single model runs with `sbatch --export=ALL,MODEL_NAME="llama3.1:8b" 2_backbone_models/job1qualfeedback_llm.sh`.

3. Stability. Each variant generates reworded transcripts, then runs the pipeline over them:
```bash
sbatch 3_rq1_stability/variant_a_all/varA_job1_qualAll.sh      # then job2, job3
sbatch 3_rq1_stability/variant_b/jobvariantb.sh                # then pipeline/job1-3
sbatch 3_rq1_stability/variant_c/jobvariantc.sh                # then pipeline/varC_job1-3
for r in 5 10 20 50; do sbatch 3_rq1_stability/variant_d/job_vard_gen$r.sh; done
```
Variant A needs no generation step. `generate_variant_d.py` takes `--rate {5,10,20,50}` and uses seeds 1 to 5.

4. Injection:
```bash
sbatch 4_rq2_injection/job_vpi_full.sh
sbatch 4_rq2_injection/job_dynamic_vpi_v2_subtle.sh
sbatch 4_rq2_injection/job_heldout_vpi.sh
```

5. Calibration. One submission per condition:
```bash
sbatch --export=ALL,CONDITION=main 5_rq3_calibration/job_rq3.sh
sbatch 5_rq3_calibration/job_rq3_explanation_first_k10.sh
```
The other conditions are `no_qual`, `no_quant` and `transcript_only`.

Every stage has analysis scripts that run on the login node once the jobs finish. They are in the `analysis/` subdirectory of each stage, or alongside the runners for stages 4 and 5.

## Code

- [1_baseline](1_baseline) contains the reproduction of the released pipeline on Gemma 3 27B, and the split manifest every other stage imports.

- [2_backbone_models](2_backbone_models) contains the same pipeline with the backbone model selected through `MODEL_NAME`.

- [3_rq1_stability](3_rq1_stability) contains the four transcript perturbations and the scripts that measure how far the output moves.

- [4_rq2_injection](4_rq2_injection) contains the injection payloads, three defended meta-review agents, and the experiment runners.

- [5_rq3_calibration](5_rq3_calibration) contains the four confidence signals and the calibration analysis.

- [scratch](scratch) contains one-off checks written while verifying results. They are not part of the pipeline.

Results are written to `analysis_output/` and generated transcripts to `rq1_perturbations/`, both under `$AIPSY_ROOT`. Scripts checkpoint after each subject, so a job that hits its wall time is restarted by resubmitting it.

## Notes

- `1_baseline/split_manifest.py` holds the train, validation and test split. Recomputing that split at runtime does not reproduce the released one: it puts participants 339 and 345 into the few-shot store they are later evaluated against. `generate_pickle.py` refuses to resume from a store containing them, and `analysis/verify_fewshot_split.py` checks it afterwards.

- `4_rq2_injection/agents/meta_reviewer.py` is the undefended agent from the released pipeline, included unmodified so the injection runs work. Every other file in that directory is original.

- `3_rq1_stability/variant_c/jobvariantc.sh` runs three subjects as a smoke test. Drop `--test_ids` and raise the wall time for the full set.

- The job scripts request `--partition=gpu-short` or `gpu-mig-40g` with `--gres=gpu:4g.40gb:1`. Partition names are site-specific and need changing on another cluster.

## References

- [AI Psychiatrist Assistant](https://github.com/trendscenter/ai-psychiatrist), the pipeline under evaluation. MIT licensed; its licence is included as [`LICENSE-upstream`](LICENSE-upstream).

- [Ollama documentation](https://github.com/ollama/ollama)

- [DAIC-WOZ](https://dcapswoz.ict.usc.edu/), distributed by USC ICT under its own licence.

## Citation

The pipeline evaluated here:

```
@InProceedings{greene26,
  title = {{AI} Psychiatrist Assistant: An {LLM}-based Multi-Agent System for Depression Assessment from Clinical Interviews},
  author = {Greene, Adam and Blair, Neviah and Mahdipour Aghabagher, Samin and Kumari, Simmi and Schlund, Michael W. and Fedorov, Alex and Calhoun, Vince D. and Li, Xinhui and Silva, Rogers F.},
  booktitle = {Proceedings of the Fifth Machine Learning for Health Symposium},
  pages = {525--542},
  year = {2026},
  volume = {297},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v297/greene26a.html}
}
```
