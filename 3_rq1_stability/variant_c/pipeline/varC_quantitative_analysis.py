# RQ1 Variant C — Stage 3: Zero-shot Quantitative Analysis
import json
import requests
import pandas as pd
from pydantic import BaseModel
from typing import List, Union
import csv
import os
from datetime import datetime

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

OLLAMA_NODE = os.environ.get("OLLAMA_NODE", "localhost")
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
model = "gemma3:27b"

dev_split_phq8   = pd.read_csv(f"{DAIC}/labels/dev_split_Depression_AVEC2017.csv")
train_split_phq8 = pd.read_csv(f"{DAIC}/labels/train_split_Depression_AVEC2017.csv")

print("NOTE: Excluded from varC pipeline — zero substitutions (identical to original): [357, 385, 423]")
participant_list = [
    316, 319, 330, 339, 345, 362, 367, 370, 375, 377, 379, 383,
    386, 389, 390, 393, 409, 413, 417, 422, 427, 428, 430,
    436, 441, 445, 447, 449, 451, 455, 456, 459, 468, 472, 484, 485,
    487, 489
]

OUTPUT_DIR = f"{ROOT}/analysis_output/VariantC/quan"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PHQ8Score(BaseModel):
    evidence: str
    reason: str
    score: Union[int, str]

class PHQ8Assessment(BaseModel):
    PHQ8_NoInterest: PHQ8Score
    PHQ8_Depressed: PHQ8Score
    PHQ8_Sleep: PHQ8Score
    PHQ8_Tired: PHQ8Score
    PHQ8_Appetite: PHQ8Score
    PHQ8_Failure: PHQ8Score
    PHQ8_Concentrating: PHQ8Score
    PHQ8_Moving: PHQ8Score

def run_phq8_analysis(transcript, ground_truth):
    prompt = f"""You are a psychiatrist evaluating a patient for depression using the PHQ-8 scale.

PHQ-8 Scoring System:
- 0 = Not at all (0-1 days)
- 1 = Several days (2-6 days)
- 2 = More than half the days (7-11 days)
- 3 = Nearly every day (12-14 days)
- N/A = Not discussed or cannot be determined from the transcript

Here is the interview transcript:
{transcript}

Ground truth PHQ-8 scores for reference:
- PHQ8_NoInterest: {ground_truth['PHQ8_NoInterest']}
- PHQ8_Depressed: {ground_truth['PHQ8_Depressed']}
- PHQ8_Sleep: {ground_truth['PHQ8_Sleep']}
- PHQ8_Tired: {ground_truth['PHQ8_Tired']}
- PHQ8_Appetite: {ground_truth['PHQ8_Appetite']}
- PHQ8_Failure: {ground_truth['PHQ8_Failure']}
- PHQ8_Concentrating: {ground_truth['PHQ8_Concentrating']}
- PHQ8_Moving: {ground_truth['PHQ8_Moving']}

Analyze the transcript and provide PHQ-8 scores. Return ONLY a JSON object with this structure:
{{
    "PHQ8_NoInterest": {{"evidence": "...", "reason": "...", "score": 0}},
    "PHQ8_Depressed":  {{"evidence": "...", "reason": "...", "score": 0}},
    "PHQ8_Sleep":      {{"evidence": "...", "reason": "...", "score": 0}},
    "PHQ8_Tired":      {{"evidence": "...", "reason": "...", "score": 0}},
    "PHQ8_Appetite":   {{"evidence": "...", "reason": "...", "score": 0}},
    "PHQ8_Failure":    {{"evidence": "...", "reason": "...", "score": 0}},
    "PHQ8_Concentrating": {{"evidence": "...", "reason": "...", "score": 0}},
    "PHQ8_Moving":     {{"evidence": "...", "reason": "...", "score": "N/A"}}
}}
Use "N/A" (string) for score when not discussed. No extra text outside the JSON."""

    response = requests.post(
        BASE_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
        },
        timeout=300
    )

    try:
        content = response.json()['message']['content']
        start = content.find('{')
        end   = content.rfind('}') + 1
        if start == -1 or end == 0:
            return None, None, None, None, None

        json_str = content[start:end]
        data = json.loads(json_str)
        phq8_scores = PHQ8Assessment(**data)

        scores = []
        gt_scores = []
        num_na = 0
        questions = ['PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired',
                     'PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating','PHQ8_Moving']

        for q in questions:
            score_obj = getattr(phq8_scores, q)
            gt_val    = ground_truth[q]
            if score_obj.score == "N/A" or score_obj.score is None:
                num_na += 1
            else:
                scores.append(int(score_obj.score))
                gt_scores.append(int(gt_val))

        if scores:
            diffs = [abs(s - g) for s, g in zip(scores, gt_scores)]
            avg_diff = sum(diffs) / len(diffs)
            acc = sum(1 for d in diffs if d == 0) / len(diffs)
        else:
            avg_diff = None
            acc = None

        overall_acc = sum(1 for s, g in zip(scores, gt_scores) if s == g) / len(questions) if scores else None

        return phq8_scores, avg_diff, acc, num_na, overall_acc

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        print(f"Error parsing response: {e}")
        return None, None, None, None, None

if __name__ == "__main__":

    csv_file  = os.path.join(OUTPUT_DIR, "results_zero_shot_varC.csv")
    json_file = os.path.join(OUTPUT_DIR, "results_zero_shot_varC_detailed.jsonl")

    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        done_ids = set(existing_df['participant_id'].tolist())
        skipped = [p for p in participant_list if p in done_ids]
        participant_list = [p for p in participant_list if p not in done_ids]
        if skipped:
            print(f"Resuming: skipping {len(skipped)} already-done: {sorted(skipped)}")
        print(f"Remaining to process: {len(participant_list)}")

    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['participant_id','timestamp',
                             'PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired',
                             'PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating','PHQ8_Moving',
                             'avg_difference','accuracy_on_available','num_questions_na','overall_accuracy'])

    for participant_id in participant_list:
        transcript_path = f"{ROOT}/rq1_perturbations/variant_c/{participant_id}_TRANSCRIPT_varC.csv"
        if not os.path.exists(transcript_path):
            print(f"Transcript not found for {participant_id} - skipping")
            continue

        current_transcript = pd.read_csv(transcript_path, sep="\t")
        current_patient_transcript = '\n'.join(
            (current_transcript['speaker'].fillna('') + ': ' + current_transcript['value'].fillna('')).tolist()
        )

        if participant_id in train_split_phq8['Participant_ID'].values:
            ground_truth = train_split_phq8[train_split_phq8['Participant_ID'] == participant_id].iloc[0]
        else:
            ground_truth = dev_split_phq8[dev_split_phq8['Participant_ID'] == participant_id].iloc[0]

        phq8_scores, avg_diff, acc, num_na, overall_acc = run_phq8_analysis(current_patient_transcript, ground_truth)

        if phq8_scores is not None:
            timestamp = datetime.now().isoformat()
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    participant_id, timestamp,
                    phq8_scores.PHQ8_NoInterest.score, phq8_scores.PHQ8_Depressed.score,
                    phq8_scores.PHQ8_Sleep.score, phq8_scores.PHQ8_Tired.score,
                    phq8_scores.PHQ8_Appetite.score, phq8_scores.PHQ8_Failure.score,
                    phq8_scores.PHQ8_Concentrating.score, phq8_scores.PHQ8_Moving.score,
                    avg_diff, acc, num_na, overall_acc
                ])

            detailed = {
                "participant_id": participant_id,
                "timestamp": timestamp,
                "PHQ8_NoInterest":    {"evidence": phq8_scores.PHQ8_NoInterest.evidence,    "reason": phq8_scores.PHQ8_NoInterest.reason,    "score": phq8_scores.PHQ8_NoInterest.score},
                "PHQ8_Depressed":     {"evidence": phq8_scores.PHQ8_Depressed.evidence,     "reason": phq8_scores.PHQ8_Depressed.reason,     "score": phq8_scores.PHQ8_Depressed.score},
                "PHQ8_Sleep":         {"evidence": phq8_scores.PHQ8_Sleep.evidence,         "reason": phq8_scores.PHQ8_Sleep.reason,         "score": phq8_scores.PHQ8_Sleep.score},
                "PHQ8_Tired":         {"evidence": phq8_scores.PHQ8_Tired.evidence,         "reason": phq8_scores.PHQ8_Tired.reason,         "score": phq8_scores.PHQ8_Tired.score},
                "PHQ8_Appetite":      {"evidence": phq8_scores.PHQ8_Appetite.evidence,      "reason": phq8_scores.PHQ8_Appetite.reason,      "score": phq8_scores.PHQ8_Appetite.score},
                "PHQ8_Failure":       {"evidence": phq8_scores.PHQ8_Failure.evidence,       "reason": phq8_scores.PHQ8_Failure.reason,       "score": phq8_scores.PHQ8_Failure.score},
                "PHQ8_Concentrating": {"evidence": phq8_scores.PHQ8_Concentrating.evidence, "reason": phq8_scores.PHQ8_Concentrating.reason, "score": phq8_scores.PHQ8_Concentrating.score},
                "PHQ8_Moving":        {"evidence": phq8_scores.PHQ8_Moving.evidence,        "reason": phq8_scores.PHQ8_Moving.reason,        "score": phq8_scores.PHQ8_Moving.score}
            }
            with open(json_file, 'a') as f:
                f.write(json.dumps(detailed) + '\n')

            print(f"Completed: {participant_id}")
        else:
            print(f"Failed: {participant_id}")

        print("="*80)
