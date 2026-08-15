# RQ1 Variant C — Stage 1: Qualitative Assessment
import json
import requests
import pandas as pd
import os
import time

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

OLLAMA_NODE = os.environ.get("OLLAMA_NODE", "localhost")
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"

print(f"DEBUG: OLLAMA_NODE = {OLLAMA_NODE}")
print(f"DEBUG: BASE_URL = {BASE_URL}")

model = "gemma3:27b"

print("NOTE: Excluded from varC pipeline — zero substitutions (identical to original): [357, 385, 423]")
VARC_TEST_IDS = [
    316, 319, 330, 339, 345, 362, 367, 370, 375, 377, 379, 383,
    386, 389, 390, 393, 409, 413, 417, 422, 427, 428, 430,
    436, 441, 445, 447, 449, 451, 455, 456, 459, 468, 472, 484, 485,
    487, 489
]

all_subjects = [(subj, 'test') for subj in VARC_TEST_IDS]

OUTPUT_DIR = f"{ROOT}/analysis_output/VariantC/qual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

checkpoint_file = os.path.join(OUTPUT_DIR, "checkpoint_qual_varC.csv")
if os.path.exists(checkpoint_file):
    checkpoint_df = pd.read_csv(checkpoint_file)
    processed_ids = set(checkpoint_df["participant_id"].tolist())
    results = checkpoint_df.to_dict("records")
    all_subjects = [(subj, dtype) for subj, dtype in all_subjects if subj not in processed_ids]
    print(f"Resuming from checkpoint: {len(processed_ids)} already done, {len(all_subjects)} remaining")
else:
    results = []
    print("No checkpoint found, starting fresh")

print(f"Total subjects to process: {len(all_subjects)}")

phq8_symptoms = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual"
]

system_prompt = "You are a psychiatrist. Your job is to assess and diagnose patients for depression. Be as objective, concise, and accurate as possible. Be specific and avoid vague generalities."

runtime_results = []
processed_count = 0
skipped_count = 0

for i, (participant_id, dataset_type) in enumerate(all_subjects):
    print(f"\n--- Processing {i+1}/{len(all_subjects)}: {participant_id} ---")
    participant_start_time = time.time()

    id_transcript = f"{ROOT}/rq1_perturbations/variant_c/{participant_id}_TRANSCRIPT_varC.csv"

    print(f"Looking for transcript at: {id_transcript}")

    if not os.path.exists(id_transcript):
        print(f"Transcript not found for {participant_id}")
        skipped_count += 1
        runtime_results.append({
            "participant_id": participant_id,
            "status": "skipped",
            "runtime_seconds": time.time() - participant_start_time
        })
        continue

    print("Transcript found, loading data...")

    try:
        transcript_data = pd.read_csv(id_transcript, sep='\t').copy()
        transcript_data['speaker'] = transcript_data['speaker'].fillna('Unknown').astype(str)
        transcript_data['value'] = transcript_data['value'].fillna('').astype(str)

        dialogue_txt = transcript_data[['speaker', 'value']].copy()
        dialogue_txt['dialogue'] = dialogue_txt['speaker'] + " : " + dialogue_txt['value']
        full_dialogue = "\n".join(dialogue_txt['dialogue'].tolist())

        print(f"Dialogue length: {len(full_dialogue)} characters")

        overall_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{full_dialogue}

Provide an overall qualitative assessment of the subject's mental health.

Please answer in this XML format. Use straight quotes instead of curly quotes, and do not include any additional text outside the XML tags:

<assessment>Summary of participant's overall mental health</assessment>
<quotes>Exact quotes from the transcript that support the assessment</quotes>
"""
        response = requests.post(
            BASE_URL,
            json={
                "model": model,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": overall_prompt}],
                "stream": False,
                "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
            },
            timeout=300
        )

        if response.status_code != 200:
            print(f"API request failed: {response.status_code}")
            runtime_results.append({"participant_id": participant_id, "status": "failed",
                                    "runtime_seconds": time.time() - participant_start_time})
            continue

        overall_content = response.json()['message']['content']

        phq8_results = []
        for symptom in phq8_symptoms:
            print(f"Assessing: {symptom[:50]}...")
            symptom_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{full_dialogue}

Assess this specific PHQ-8 symptom: {symptom}

For this symptom, note:
- Frequency (daily, several days, more than half the days, nearly every day)
- Duration (how long experienced)
- Severity/impact on functioning
- If not discussed, state "not assessed in interview"

Please answer in this XML format. Use straight quotes instead of curly quotes:

<assessment>Summary of this symptom's presence, frequency, duration, and severity</assessment>
<quotes>Exact quotes from the transcript that support this assessment</quotes>
"""
            response = requests.post(
                BASE_URL,
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": system_prompt},
                                 {"role": "user", "content": symptom_prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
                },
                timeout=300
            )
            if response.status_code != 200:
                phq8_results.append(f"ERROR: Failed to assess symptom")
                continue
            phq8_results.append(f"**{symptom}**\n{response.json()['message']['content']}")

        print("Assessing social factors...")
        social_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{full_dialogue}

Summarize social aspects that may influence the subject's mental health (ex. familial relationships, friendship dynamics, work environment, etc.).

Please answer in this XML format. Use straight quotes instead of curly quotes:

<assessment>Summary of social influences on patient's health</assessment>
<quotes>Quotes from the transcript that support the assessment</quotes>
"""
        response = requests.post(BASE_URL, json={"model": model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": social_prompt}],
            "stream": False, "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}},
            timeout=300)
        social_content = response.json()['message']['content'] if response.status_code == 200 else "ERROR"

        print("Assessing biological factors...")
        biological_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{full_dialogue}

Summarize biological aspects that may influence the subject's mental health (ex. familial history of mental health issues, previous or pre-existing mental health issues, stress levels, etc.).

Please answer in this XML format. Use straight quotes instead of curly quotes:

<assessment>Summary of biological influences on patient's health</assessment>
<quotes>Quotes from the transcript that support the assessment</quotes>
"""
        response = requests.post(BASE_URL, json={"model": model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": biological_prompt}],
            "stream": False, "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}},
            timeout=300)
        biological_content = response.json()['message']['content'] if response.status_code == 200 else "ERROR"

        print("Assessing risk factors...")
        risk_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{full_dialogue}

Identify potential risk factors the subject may be experiencing.

Please answer in this XML format. Use straight quotes instead of curly quotes:

<assessment>Summary of potential risk factors</assessment>
<quotes>Exact quotes from the transcript that support the assessment</quotes>
"""
        response = requests.post(BASE_URL, json={"model": model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": risk_prompt}],
            "stream": False, "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}},
            timeout=300)
        risk_content = response.json()['message']['content'] if response.status_code == 200 else "ERROR"

        final_assessment = f"""=== OVERALL ASSESSMENT ===
{overall_content}

=== PHQ-8 SYMPTOMS ===
{chr(10).join(phq8_results)}

=== SOCIAL FACTORS ===
{social_content}

=== BIOLOGICAL FACTORS ===
{biological_content}

=== RISK FACTORS ===
{risk_content}
"""
        runtime_seconds = time.time() - participant_start_time
        print(f"Runtime: {runtime_seconds:.2f}s")

        results.append({
            "participant_id": participant_id,
            "dataset_type": dataset_type,
            "qualitative_assessment": final_assessment
        })
        runtime_results.append({"participant_id": participant_id, "status": "success",
                                 "runtime_seconds": runtime_seconds})
        processed_count += 1

        if len(results) == 1 or len(results) % 10 == 0 or len(results) == len(all_subjects):
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)
            print(f"Checkpoint saved: {len(results)} subjects → {checkpoint_file}")
            pd.DataFrame(runtime_results).to_csv(
                os.path.join(OUTPUT_DIR, "checkpoint_runtime_varC.csv"), index=False)

    except Exception as e:
        print(f"Error processing {participant_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        runtime_results.append({"participant_id": participant_id, "status": "error",
                                 "runtime_seconds": time.time() - participant_start_time})
        continue

print(f"\nProcessed: {processed_count} | Skipped: {skipped_count}")

if results:
    out = os.path.join(OUTPUT_DIR, "qual_assessment_varC.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Final qual output: {out}")

if runtime_results:
    rt = os.path.join(OUTPUT_DIR, "runtime_varC.csv")
    pd.DataFrame(runtime_results).to_csv(rt, index=False)
    print(f"Runtime log: {rt}")
