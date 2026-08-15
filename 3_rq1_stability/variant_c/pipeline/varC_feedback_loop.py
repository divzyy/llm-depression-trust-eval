# RQ1 Variant C — Stage 2: Feedback Loop
import os
import pandas as pd
import requests
import time
import re

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

def parse_score_and_explanation(response_text):
    score_patterns = [
        r'score[:\s]*(\d+)',
        r'(\d+)[/\s]*(?:out of\s*)?5',
        r'(\d+)[/\s]*5',
        r'rating[:\s]*(\d+)',
        r'^(\d+)',
    ]
    score = None
    for pattern in score_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            potential_score = int(match.group(1))
            if 1 <= potential_score <= 5:
                score = potential_score
                break
    return score, response_text.strip()

OLLAMA_NODE = os.environ.get("OLLAMA_NODE", "localhost")
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
model = "gemma3:27b"

OUTPUT_DIR = f"{ROOT}/analysis_output/VariantC/qual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

input_csv_path          = os.path.join(OUTPUT_DIR, "qual_assessment_varC.csv")
feedback_assessments_csv = os.path.join(OUTPUT_DIR, "feedback_assessments_varC.csv")
feedback_evaluations_csv = os.path.join(OUTPUT_DIR, "feedback_evaluations_varC.csv")

print(f"Input:               {input_csv_path}")
print(f"Feedback assessments: {feedback_assessments_csv}")
print(f"Feedback evaluations: {feedback_evaluations_csv}")

df = pd.read_csv(input_csv_path)
print(f"Loaded {len(df)} participants")

feedback_assessments = []
feedback_evaluations = []
processed_count = 0
skipped_count = 0
feedback_count = 0
failed_evaluations = []

completed_subjects = set()
if os.path.exists(feedback_assessments_csv):
    existing = pd.read_csv(feedback_assessments_csv)
    completed_subjects.update(existing['participant_id'].tolist())
    feedback_assessments = existing.to_dict('records')
    print(f"Found existing feedback assessments: {len(feedback_assessments)}")

if os.path.exists(feedback_evaluations_csv):
    existing = pd.read_csv(feedback_evaluations_csv)
    completed_subjects.update(existing['participant_id'].tolist())
    feedback_evaluations = existing.to_dict('records')
    print(f"Found existing feedback evaluations: {len(feedback_evaluations)}")

if completed_subjects:
    df = df[~df['participant_id'].isin(completed_subjects)]
    print(f"Remaining to process: {len(df)}")

for index, row in df.iterrows():
    participant_id = row['participant_id']
    qualitative_assessment = str(row['qualitative_assessment']) if 'qualitative_assessment' in row else ""
    original_dataset_type = str(row['dataset_type']) if 'dataset_type' in row else "test"

    print(f"\n--- Processing {index+1}/{len(df)}: {participant_id} ---")

    if qualitative_assessment.startswith('```xml'):
        qualitative_assessment = qualitative_assessment[6:]
    if qualitative_assessment.endswith('```'):
        qualitative_assessment = qualitative_assessment[:-3]
    qualitative_assessment = qualitative_assessment.strip()
    if len(qualitative_assessment) > 3000:
        qualitative_assessment = qualitative_assessment[:3000] + "..."
    qualitative_assessment = re.sub(r'<[^>]+>', '', qualitative_assessment)
    qualitative_assessment = re.sub(r'\s+', ' ', qualitative_assessment)

    id_transcript = f"{ROOT}/rq1_perturbations/variant_c/{participant_id}_TRANSCRIPT_varC.csv"
    if not os.path.exists(id_transcript):
        print(f"Transcript not found for {participant_id}")
        skipped_count += 1
        continue

    try:
        transcript_data = pd.read_csv(id_transcript, sep='\t').copy()
        transcript_data['speaker'] = transcript_data['speaker'].fillna('Unknown').astype(str)
        transcript_data['value'] = transcript_data['value'].fillna('').astype(str)
        dialogue_txt = transcript_data[['speaker', 'value']].copy()
        dialogue_txt['dialogue'] = dialogue_txt['speaker'] + " : " + dialogue_txt['value']
        transcript = "\n".join(dialogue_txt['dialogue'].tolist())
    except Exception as e:
        print(f"Error loading transcript: {e}")
        skipped_count += 1
        continue

    start_time = time.time()

    coherence_prompt = f"""Evaluate the following qualitative assessment output for COHERENCE only. Compare qualitative_assessment to the provided transcript.
Score from 1-5 solely based on the following criteria:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
*NO FLOATING POINT SCORES, ONLY INTEGERS 1-5*
Coherence (1-5): Is the response logically consistent? Mistakes are logically inconsistent statements or contradictions.

Format your response as:
Score: [your score]
Explanation: [explanation]

---
Here is the transcript:
{transcript}

Here is the assessment:
{qualitative_assessment}
---"""

    completeness_prompt = f"""Evaluate for COMPLETENESS only.
Score from 1-5:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
Completeness (1-5): Does the assessment cover all relevant symptoms, severities, duration/frequency?

Format: Score: [score] / Explanation: [explanation]

Transcript: {transcript}
Assessment: {qualitative_assessment}"""

    specificity_prompt = f"""Evaluate for SPECIFICITY only.
Score from 1-5:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
Specificity (1-5): Is the assessment specific? Mistakes include vague statements like 'the patient seems depressed'.

Format: Score: [score] / Explanation: [explanation]

Transcript: {transcript}
Assessment: {qualitative_assessment}"""

    accuracy_prompt = f"""Evaluate for ACCURACY only.
Score from 1-5:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
Accuracy (1-5): Are symptoms aligned with DSM-5 or PHQ-8?

Format: Score: [score] / Explanation: [explanation]

Transcript: {transcript}
Assessment: {qualitative_assessment}"""

    timeout = 300

    try:
        initial_scores = {}
        initial_explanations = {}

        for metric, prompt in [('coherence', coherence_prompt), ('completeness', completeness_prompt),
                                ('specificity', specificity_prompt), ('accuracy', accuracy_prompt)]:
            print(f"  Getting {metric} score...")
            resp = requests.post(BASE_URL, json={"model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False, "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}},
                timeout=timeout-10)
            if resp.status_code == 200:
                content = resp.json()['message']['content']
                score, _ = parse_score_and_explanation(content)
                initial_scores[metric] = score
                initial_explanations[metric] = content
                print(f"  {metric}: {score}")
            else:
                initial_scores[metric] = None
                initial_explanations[metric] = None
            time.sleep(2)

        low_scores = [m for m in ['coherence','completeness','specificity','accuracy']
                      if initial_scores.get(m) and initial_scores[m] <= 2]

        if low_scores:
            print(f"  Low scores: {low_scores} — running feedback loop...")
            feedback_count += 1
            iteration = 0
            max_iterations = 10
            current_assessment = qualitative_assessment
            current_scores = initial_scores.copy()
            current_explanations = initial_explanations.copy()

            system_prompt = "You are a psychiatrist. Your job is to assess and diagnose patients for depression. Be as objective, concise, and accurate as possible. Be specific and avoid vague generalities."

            while low_scores and iteration < max_iterations:
                iteration += 1
                print(f"  Iteration {iteration}: improving {low_scores}")

                feedback_details = []
                metric_desc = {
                    'coherence': 'logical consistency',
                    'completeness': 'comprehensive coverage',
                    'specificity': 'specific details',
                    'accuracy': 'accuracy per DSM-5/PHQ-8'
                }
                for metric in low_scores:
                    if current_scores[metric] and current_explanations[metric]:
                        feedback_details.append(
                            f"- **{metric.upper()}** scored {current_scores[metric]}/5: "
                            f"{current_explanations[metric]}\n  Improve: {metric_desc[metric]}"
                        )

                for section, section_prompt in [
                    ("overall", f"""Improve assessment addressing feedback:\n{chr(10).join(feedback_details)}\n\nTRANSCRIPT:\n{transcript}\n\nProvide improved qualitative assessment.\n\nXML format:\n<assessment>Summary</assessment>\n<quotes>Quotes</quotes>"""),
                    ("social",  f"""Improve social factors addressing feedback:\n{chr(10).join(feedback_details)}\n\nTRANSCRIPT:\n{transcript}\n\n<assessment>Social influences</assessment>\n<quotes>Quotes</quotes>"""),
                    ("biological", f"""Improve biological factors addressing feedback:\n{chr(10).join(feedback_details)}\n\nTRANSCRIPT:\n{transcript}\n\n<assessment>Biological influences</assessment>\n<quotes>Quotes</quotes>"""),
                    ("risk", f"""Improve risk factors addressing feedback:\n{chr(10).join(feedback_details)}\n\nTRANSCRIPT:\n{transcript}\n\n<assessment>Risk factors</assessment>\n<quotes>Quotes</quotes>""")
                ]:
                    resp = requests.post(BASE_URL, json={"model": model,
                        "messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": section_prompt}],
                        "stream": False, "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}},
                        timeout=timeout)
                    if resp.status_code == 200:
                        globals()[f"{section}_content"] = resp.json()['message']['content']
                    else:
                        globals()[f"{section}_content"] = "ERROR"

                current_assessment = f"Overall:\n{overall_content}\nSocial:\n{social_content}\nBiological:\n{biological_content}\nRisk:\n{risk_content}"

                for metric in low_scores:
                    time.sleep(2)
                    prompt_map = {'coherence': coherence_prompt, 'completeness': completeness_prompt,
                                  'specificity': specificity_prompt, 'accuracy': accuracy_prompt}
                    new_prompt = prompt_map[metric].replace(qualitative_assessment, current_assessment)
                    resp = requests.post(BASE_URL, json={"model": model,
                        "messages": [{"role": "user", "content": new_prompt}],
                        "stream": False, "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}},
                        timeout=timeout-10)
                    if resp.status_code == 200:
                        content = resp.json()['message']['content']
                        score, _ = parse_score_and_explanation(content)
                        current_scores[metric] = score
                        current_explanations[metric] = content
                        print(f"    {metric} re-eval: {score}")

                low_scores = [m for m in ['coherence','completeness','specificity','accuracy']
                              if current_scores.get(m) and current_scores[m] <= 2]

            feedback_assessments.append({
                'participant_id': participant_id,
                'dataset_type': original_dataset_type,
                'qualitative_assessment': current_assessment
            })
            feedback_evaluations.append({
                'participant_id': participant_id,
                'coherence': current_scores.get('coherence'),
                'coherence_explanation': current_explanations.get('coherence'),
                'completeness': current_scores.get('completeness'),
                'completeness_explanation': current_explanations.get('completeness'),
                'specificity': current_scores.get('specificity'),
                'specificity_explanation': current_explanations.get('specificity'),
                'accuracy': current_scores.get('accuracy'),
                'accuracy_explanation': current_explanations.get('accuracy')
            })
            processed_count += 1
        else:
            print(f"  All scores OK — no feedback needed")

        elapsed = time.time() - start_time
        print(f"  Done in {elapsed:.1f}s")

    except Exception as e:
        print(f"Error: {e}")
        failed_evaluations.append(participant_id)

    if (len(feedback_assessments) % 10 == 0 and len(feedback_assessments) > 0) or len(feedback_assessments) == 1:
        if feedback_assessments:
            pd.DataFrame(feedback_assessments).to_csv(feedback_assessments_csv, index=False)
        if feedback_evaluations:
            pd.DataFrame(feedback_evaluations).to_csv(feedback_evaluations_csv, index=False)

    time.sleep(1)

if feedback_assessments:
    pd.DataFrame(feedback_assessments).to_csv(feedback_assessments_csv, index=False)
    print(f"Saved: {feedback_assessments_csv}")
if feedback_evaluations:
    pd.DataFrame(feedback_evaluations).to_csv(feedback_evaluations_csv, index=False)
    print(f"Saved: {feedback_evaluations_csv}")

print(f"\n=== SUMMARY ===")
print(f"Feedback loops applied: {feedback_count}")
print(f"Processed: {processed_count} | Skipped: {skipped_count}")
print(f"Failed: {failed_evaluations}")
