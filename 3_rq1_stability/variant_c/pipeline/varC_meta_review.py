# RQ1 Variant C — Stage 5: Meta Review (few-shot quantitative input)
import os
import csv
import json
import requests
import pandas as pd

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_response(response_text, key_list):
    output_dict = {}
    for k in key_list:
        try:
            idx1 = response_text.index(f"<{k}>")
            idx2 = response_text.index(f"</{k}>")
            output_dict[k] = response_text[idx1 + len(f"<{k}>") : idx2]
        except ValueError:
            output_dict[k] = "N/A"
    return output_dict

def main():
    OLLAMA_NODE = os.environ.get("OLLAMA_NODE", "localhost")
    BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
    model = "gemma3:27b"

    rootdir = f"{ROOT}"

    OUTPUT_DIR = os.path.join(rootdir, "analysis_output/VariantC/meta")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_filename = os.path.join(OUTPUT_DIR, "meta_review_fewshot_varC.csv")

    qual_df = pd.read_csv(os.path.join(rootdir, "analysis_output/VariantC/qual/qual_assessment_varC.csv"))

    quan_list = load_jsonl(os.path.join(
        rootdir,
        "analysis_output/VariantC/quan_few_shot/ids_test_chunk_8_step_2_dim_4096_examples_2_embedding_results_analysis_1.jsonl"
    ))

    phq8_questions = ['PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired',
                      'PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating','PHQ8_Moving']

    quan_list_ = []
    for i in quan_list:
        quan_dict = {"participant_id": i["participant_id"]}
        for j in phq8_questions:
            quan_dict[j]            = i[j]["score"]
            quan_dict[f"{j}_Reason"] = i[j]["reason"]
        quan_list_.append(quan_dict)

    quan_df = pd.DataFrame(quan_list_)
    participant_id_list = sorted(quan_df["participant_id"].tolist())

    output_list = []

    for participant_id in participant_id_list:
        transcript_path = f"{ROOT}/rq1_perturbations/variant_c/{participant_id}_TRANSCRIPT_varC.csv"
        if not os.path.exists(transcript_path):
            print(f"Transcript missing for {participant_id}, skipping")
            continue

        raw_transcript = pd.read_csv(transcript_path, sep='\t')
        raw_transcript['speaker'] = raw_transcript['speaker'].fillna('Unknown').astype(str)
        raw_transcript['value']   = raw_transcript['value'].fillna('').astype(str)
        transcript = '\n'.join(raw_transcript['speaker'] + ': ' + raw_transcript['value'])

        qual_rows = qual_df.loc[qual_df['participant_id'] == participant_id, 'qualitative_assessment']
        if qual_rows.empty:
            print(f"Qual assessment missing for {participant_id}, skipping")
            continue
        qual_output = qual_rows.iloc[0][7:-4]

        ind = quan_df['participant_id'] == participant_id
        quan_output = ""
        for i in phq8_questions:
            score  = quan_df.loc[ind, i].iloc[0]
            reason = quan_df.loc[ind, f"{i}_Reason"].iloc[0]
            if score != "N/A":
                quan_output += f"<{i.lower()}_score>{score}</{i.lower()}_score>\n"
                quan_output += f"<{i.lower()}_explanation>{reason}</{i.lower()}_explanation>\n"

        system_prompt = "You are an AI psychiatrist assistant specializing in depression. Your task is to analyze the interview transcript and related qualitative and quantitative assessments, and then predict the diagnosis label and severity level."

        meta_prompt = f"""Please review a participant's interview transcript, qualitative assessment, and quantitative assessment below.

        Here is the interview transcript in <transcript> tags:
        <transcript>
        {transcript}
        </transcript>

        Here are the overall assessment, social and biological risk factors in <qualitative_assessment> tags:
        <qualitative_assessment>
        {qual_output}
        </qualitative_assessment>

        Here are the predicted PHQ-8 scores and explanations in <quantitative_assessment> tags:
        <quantitative_assessment>\n{quan_output}</quantitative_assessment>

        According to the PHQ-8: 0-4 = no significant symptoms; 5-9 = mild; 10-14 = moderate; 15-19 = moderately severe; 20-24 = severe.
        Note the interview may not cover all eight PHQ-8 items. Use available information to infer and predict accurately.
        Predict severity in <severity> tags (0=none, 1=mild, 2=moderate, 3=moderately severe, 4=severe).
        Explain in <explanation> tags. Answer in XML format with each tag on a new line.
        """

        response = requests.post(
            BASE_URL,
            json={
                "model": model,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": meta_prompt}],
                "stream": False,
                "options": {"temperature": 0, "top_k": 20, "top_p": 1}
            },
            timeout=1800
        )

        try:
            response_text = response.json()["message"]["content"]
            print(f"{participant_id}: {response_text[:100]}")
        except KeyError:
            print(f"{participant_id}: error")
            continue

        prediction_dict = extract_response(response_text, ["severity", "explanation"])
        output_dict = {"participant_id": participant_id, "response": response_text}
        output_dict.update(prediction_dict)
        output_list.append(output_dict)

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ["participant_id", "response", "severity", "explanation"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_list)

    print(f"\nMeta review saved: {output_filename}")

if __name__ == "__main__":
    main()
