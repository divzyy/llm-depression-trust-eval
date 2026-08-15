
import os
import re
import json
import math
import csv
import requests
import pandas as pd

OLLAMA_NODE = os.environ.get("OLLAMA_NODE", "localhost")
CHAT_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
MODEL = "gemma3:27b"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.environ.get("ROOTDIR", os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
DATA_DIR = os.environ.get("DAIC_WOZ_DATA", os.path.abspath(os.path.join(ROOTDIR, "..", "daic_woz_data")))
QUAL_PATH = os.path.join(ROOTDIR, "analysis_output", "Baseline", "qual", "qual_assessment_GEMMA_v2.csv")
QUAN_JSONL = os.path.join(ROOTDIR, "analysis_output/quan/results_zero_shot_test41_detailed.jsonl")
OUT_DIR = os.path.join(ROOTDIR, "analysis_output/rq3")
OUT_CSV = os.path.join(OUT_DIR, "pretest_wrongcases.csv")


SUBJECTS = [
    (316, "WRONG - False Positive"),
    (385, "WRONG - False Positive"),
    (386, "WRONG - False Negative"),
    (413, "WRONG - False Negative"),
    (330, "CORRECT - contrast"),
    (357, "CORRECT - contrast"),
]

N_SAMPLES = 5
SAMPLE_TEMP = 0.7

PHQ8_KEYS = [
    'PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired',
    'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving'
]

def load_ground_truth():
    """PHQ8 totals from AVEC2017 label files. Returns {pid: total} or {}."""
    gt = {}
    for fname in ["train_split_Depression_AVEC2017.csv", "dev_split_Depression_AVEC2017.csv"]:
        path = os.path.join(DATA_DIR, "labels", fname)
        if not os.path.exists(path):
            print(f"[WARN] labels file not found: {path}")
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            pid = int(row['Participant_ID'])
            if 'PHQ8_Score' in df.columns and pd.notna(row.get('PHQ8_Score')):
                gt[pid] = float(row['PHQ8_Score'])
            else:
                cols = [c for c in PHQ8_KEYS if c in df.columns]
                if cols:
                    gt[pid] = float(sum(row[c] for c in cols if pd.notna(row.get(c))))
    return gt

def load_transcript(pid):
    path = os.path.join(DATA_DIR, f"transcripts/{pid}_TRANSCRIPT.csv")
    if not os.path.exists(path):
        return None
    raw = pd.read_csv(path, sep='\t')
    raw['speaker'] = raw['speaker'].fillna('').astype(str)
    raw['value'] = raw['value'].fillna('').astype(str)
    return '\n'.join(raw['speaker'] + ': ' + raw['value'])

def load_qual(pid):
    try:
        df = pd.read_csv(QUAL_PATH)
        row = df[df['participant_id'].astype(str) == str(pid)]
        if row.empty:
            return None
        raw = row['qualitative_assessment'].iloc[0]
        return raw[7:-4] if len(raw) > 11 else raw
    except Exception as e:
        print(f"[WARN] qual load failed for {pid}: {e}")
        return None

def load_quan(pid):
    """Build the quant XML block from the saved zero-shot jsonl.
    Robust to int/str participant_id mismatch."""
    if not os.path.exists(QUAN_JSONL):
        print(f"[WARN] quant jsonl not found: {QUAN_JSONL}")
        return None
    try:
        with open(QUAN_JSONL) as f:
            for line in f:
                e = json.loads(line)
                if str(e.get("participant_id")) != str(pid):
                    continue
                out = []
                for k in PHQ8_KEYS:
                    if k not in e:
                        continue
                    score = e[k].get("score")
                    reason = e[k].get("reason", "")
                    if score != "N/A" and score is not None:
                        out.append(f"<{k.lower()}_score>{score}</{k.lower()}_score>")
                        out.append(f"<{k.lower()}_explanation>{reason}</{k.lower()}_explanation>")
                return '\n'.join(out) if out else None
    except Exception as e:
        print(f"[WARN] quant load failed for {pid}: {e}")
    return None

META_SYSTEM = ("You are an AI psychiatrist assistant specializing in depression. "
               "Your task is to analyze the interview transcript and related qualitative "
               "and quantitative assessments, and then predict the diagnosis label and severity level.")

def base_meta_prompt(transcript, qual, quan):
    return f"""Please review a participant's interview transcript, qualitative assessment, and quantitative assessment below.  

        Here is the interview transcript in <transcript> tags:
        <transcript>
        {transcript}
        </transcript>

        Here are the overall assessment, social and biological risk factors that may influence the participant's mental health in <qualitative_assessment> tags:
        <qualitative_assessment>
        {qual}
        </qualitative_assessment>

        Here are the predicted PHQ-8 scores and explanations based on the available information in the interview transcript in <quantitative_assessment> tags:
        <quantitative_assessment>\n{quan}</quantitative_assessment>

        According to the Patient Health Questionnaire eight-item depression scale (PHQ-8), a total PHQ-8 score of 0 to 4 represents no significant depressive symptoms; a total score of 5 to 9 represents mild depressive symptoms; 10 to 14, moderate; 15 to 19, moderately severe; and 20 to 24, severe. \
        Note that the interview may not cover all eight PHQ-8 items. Therefore, directly summing the available scores and deriving a severity level could underestimate the participant's condition. Please use the available information to infer and predict the participant's condition as accurately as possible. \
        Please predict their severity level in <severity> tags, where 0 means no significant depressive symptoms, 1 means mild symptoms, 2 means moderate symptoms, 3 means moderately severe symptoms, and 4 means severe symptoms. \
        Please explain your predictions in <explanation> tags. Please provide answers in the XML format with each tag on a new line.
        """

def stops_meta_prompt(transcript, qual, quan):
    
    return base_meta_prompt(transcript, qual, quan) + """
        After predicting the severity, reason carefully about your confidence in the following structured steps:
        1. In <supporting_evidence> tags: list the specific evidence from the transcript and assessments that supports your severity prediction.
        2. In <uncertainty_factors> tags: list what is missing, ambiguous, or conflicting (e.g., symptoms not discussed, contradictory signals, sparse transcript, disagreement between qualitative and quantitative assessments).
        3. In <confidence> tags: given steps 1 and 2, provide a single integer from 0 to 100 representing how confident you are that your severity prediction is correct (0 = no confidence, 100 = completely certain).
        Provide all tags in XML format, each on a new line.
        """

def chat(prompt, options, logprobs=False, timeout=1800):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": META_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": options,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 20
    r = requests.post(CHAT_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def extract_tag(text, tag):
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None

def parse_severity(text):
    sev = extract_tag(text, "severity")
    if sev is None:
        return None
    m = re.search(r"[0-4]", sev)
    return int(m.group()) if m else None

def to_binary(severity):
    """0-1 -> 0 (not depressed); 2-4 -> 1 (depressed)."""
    if severity is None:
        return None
    return 1 if severity >= 2 else 0

def locate_severity_logprob(data):
    
    lp = data.get("logprobs")
    if not lp:
        return None, None, {}
    token_list = lp.get("content") or lp.get("tokens") or [] if isinstance(lp, dict) else lp

    toks, full, char_to_tok = [], "", []
    for i, t in enumerate(token_list):
        tk = {"i": i, "token": t.get("token", ""), "logprob": t.get("logprob"),
              "top": t.get("top_logprobs", [])}
        toks.append(tk)
        for _ in tk["token"]:
            char_to_tok.append(i)
        full += tk["token"]

    idx = full.find("<severity>")
    if idx == -1:
        return None, None, {}
    j = idx + len("<severity>")
    digit_pos = -1
    while j < len(full):
        if full[j] in "01234":
            digit_pos = j
            break
        if full[j] == "<":
            break
        j += 1
    if digit_pos == -1:
        return None, None, {}

    tk = toks[char_to_tok[digit_pos]]
    digit = tk["token"].strip()
    prob = math.exp(tk["logprob"]) if tk["logprob"] is not None else None
    dist = {}
    for entry in tk["top"]:
        tt = entry.get("token", "").strip()
        if tt in ("0", "1", "2", "3", "4"):
            dist[tt] = round(math.exp(entry["logprob"]), 4)
    return digit, prob, dist

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gt_totals = load_ground_truth()
    if gt_totals:
        print(f"Ground truth loaded for {len(gt_totals)} subjects (from AVEC label files)")
    else:
        print("[WARN] No ground truth loaded - interpret results using your known FP/FN labels")

    results = []

    for pid, status in SUBJECTS:
        print("\n" + "#" * 70)
        print(f"# Participant {pid}  [{status}]")
        print("#" * 70)

        transcript = load_transcript(pid)
        if transcript is None:
            print(f"[SKIP] no transcript for {pid}")
            continue
        qual = load_qual(pid)
        quan = load_quan(pid)
        print(f"transcript: {len(transcript)} chars | qual: {'OK' if qual else 'MISSING'} | quant: {'OK' if quan else 'MISSING'}")
        if qual is None:
            qual = "(no qualitative assessment available)"
        if quan is None:
            quan = "(no quantitative assessment available)"

        gt_total = gt_totals.get(pid)
        gt_label = (1 if gt_total >= 10 else 0) if gt_total is not None else None
        print(f"GT PHQ8 total: {gt_total}  ->  GT binary: {gt_label}")

        prompt = base_meta_prompt(transcript, qual, quan)

        # ── S1: temp-0 logprob run ──
        print("\n--- S1: token logprob (temp 0) ---")
        try:
            data = chat(prompt, {"temperature": 0, "top_k": 20, "top_p": 1}, logprobs=True)
            content = data["message"]["content"]
            s1_sev = parse_severity(content)
            digit, s1_prob, dist = locate_severity_logprob(data)
            if s1_prob is not None:
                print(f"severity: {s1_sev} | located token: '{digit}' prob: {s1_prob:.4f}")
            else:
                print(f"severity: {s1_sev} | logprob locate FAILED")
            print(f"distribution over digits: {dist}")
        except Exception as e:
            print(f"[ERROR] S1 failed: {e}")
            s1_sev, s1_prob, dist = None, None, {}

        s1_binary = to_binary(s1_sev)
        s1_correct = (s1_binary == gt_label) if (s1_binary is not None and gt_label is not None) else None

        # ── S2: SToPS verbalized (temp 0) ──
        print("\n--- S2: SToPS verbalized confidence (temp 0) ---")
        try:
            data = chat(stops_meta_prompt(transcript, qual, quan),
                        {"temperature": 0, "top_k": 20, "top_p": 1})
            content = data["message"]["content"]
            s2_sev = parse_severity(content)
            conf_raw = extract_tag(content, "confidence")
            m = re.search(r"\d+", conf_raw or "")
            s2_conf = min(100, max(0, int(m.group()))) if m else None
            unc = extract_tag(content, "uncertainty_factors")
            print(f"severity: {s2_sev} | verbalized confidence: {s2_conf}")
            if unc:
                print(f"uncertainty factors (first 300 chars): {unc[:300]}")
            if s2_sev != s1_sev:
                print(f"[NOTE] S2 severity ({s2_sev}) differs from S1 severity ({s1_sev}) - prompt change shifted the prediction")
        except Exception as e:
            print(f"[ERROR] S2 failed: {e}")
            s2_sev, s2_conf = None, None

        # ── S3: sampling consistency (temp 0.7, 5 seeds) ──
        print(f"\n--- S3: sampling consistency ({N_SAMPLES} samples @ temp {SAMPLE_TEMP}) ---")
        sample_sevs = []
        for k in range(N_SAMPLES):
            try:
                data = chat(prompt, {"temperature": SAMPLE_TEMP, "top_k": 20,
                                     "top_p": 1, "seed": 1001 + k})
                sev = parse_severity(data["message"]["content"])
                sample_sevs.append(sev)
                print(f"  sample {k+1}: severity {sev}")
            except Exception as e:
                print(f"  sample {k+1}: [ERROR] {e}")
                sample_sevs.append(None)

        valid = [s for s in sample_sevs if s is not None]
        if valid and s1_binary is not None:
            agree_binary = sum(1 for s in valid if to_binary(s) == s1_binary) / len(valid)
        else:
            agree_binary = None
        if valid and s1_sev is not None:
            agree_exact = sum(1 for s in valid if s == s1_sev) / len(valid)
        else:
            agree_exact = None
        print(f"agreement with temp-0 prediction: binary={agree_binary}, exact severity={agree_exact}")

        results.append({
            "pid": pid, "status": status,
            "gt_phq8_total": gt_total, "gt_binary": gt_label,
            "s1_severity": s1_sev, "s1_binary": s1_binary, "s1_correct": s1_correct,
            "s1_token_prob": round(s1_prob, 4) if s1_prob else None,
            "s2_severity": s2_sev, "s2_confidence": s2_conf,
            "s3_samples": "|".join(str(s) for s in sample_sevs),
            "s3_agree_binary": agree_binary, "s3_agree_exact": agree_exact,
        })

    print("\n\n" + "=" * 100)
    print("SUMMARY - THE TABLE THAT DECIDES RQ3 FRAMING")
    print("=" * 100)
    hdr = (f"{'PID':<5}{'Status':<28}{'GT':<4}{'Sev':<5}{'OK?':<7}"
           f"{'S1 prob':<10}{'S2 conf':<9}{'S3 agr(bin)':<13}{'S3 samples':<15}")
    print(hdr)
    print("-" * 100)
    for r in results:
        print(f"{r['pid']:<5}{r['status']:<28}"
              f"{str(r['gt_binary']):<4}{str(r['s1_severity']):<5}"
              f"{str(r['s1_correct']):<7}"
              f"{str(r['s1_token_prob']):<10}{str(r['s2_confidence']):<9}"
              f"{str(r['s3_agree_binary']):<13}{r['s3_samples']:<15}")
    print("-" * 100)
    print("""
READ IT LIKE THIS:
  For each signal, compare its values on WRONG rows vs CORRECT rows.
  - A signal is USEFUL if it is clearly LOWER on the wrong cases.
  - A signal is USELESS if it is ~equal (e.g. always ~1.0 / always ~90) everywhere.
  S1 useful?  -> token logprobs carry real uncertainty even in the pipeline.
  S2 useful?  -> the model can verbalize doubt when asked (SToPS works).
  S3 useful?  -> sampling exposes instability on the borderline cases.
  ALL flat?   -> strong 'pipeline launders uncertainty' finding; framing shifts
                 to demonstrating that NO cheap signal flags errors.
""")

    if results:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
