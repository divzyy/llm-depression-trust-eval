

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
FEEDBACK_CANDIDATES = [
    os.path.join(ROOTDIR, "analysis_output/Baseline/qual/feedback_evaluations_v2.csv"),
    os.path.join(ROOTDIR, "analysis_output/Baseline/qual/feedback_assessments_v2.csv"),
    os.path.join(ROOTDIR, "analysis_output/qual/feedback_evaluations_v2.csv"),
]
OUT_DIR = os.path.join(ROOTDIR, "analysis_output/rq3")
QUAN_BASELINE_JSONL = os.path.join(ROOTDIR, "analysis_output/Baseline/quan/results_zero_shot_test41_detailed.jsonl")
META_BASELINE_CSV = os.path.join(ROOTDIR, "analysis_output/Baseline/qual/meta_review_zeroshot_test_v2.csv")

SUBJECTS = [
    (316, "original FP"),
    (386, "original FN"),
    (330, "wrong case (GT=12, FN)"),
    (339, "correct contrast"),
]

N_SAMPLES = 5
SAMPLE_TEMP = 0.7

PHQ8_KEYS = [
    'PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired',
    'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving'
]

def load_ground_truth():
    """Returns {pid: {'total': float, 'items': {PHQ8_*: float}}}"""
    gt = {}
    for fname in ["train_split_Depression_AVEC2017.csv", "dev_split_Depression_AVEC2017.csv"]:
        path = os.path.join(DATA_DIR, "labels", fname)
        if not os.path.exists(path):
            print(f"[WARN] labels file not found: {path}")
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            pid = int(row['Participant_ID'])
            items = {}
            for k in PHQ8_KEYS:
                if k in df.columns and pd.notna(row.get(k)):
                    items[k] = float(row[k])
            if 'PHQ8_Score' in df.columns and pd.notna(row.get('PHQ8_Score')):
                total = float(row['PHQ8_Score'])
            else:
                total = sum(items.values()) if items else None
            gt[pid] = {'total': total, 'items': items}
    return gt

def load_saved_quant_scores(pid):
    """Loads baseline quant scores from the saved jsonl, for cross-checking
    that the regenerated (temp 0, top_k 1) run matches the official baseline."""
    if not os.path.exists(QUAN_BASELINE_JSONL):
        return None
    try:
        with open(QUAN_BASELINE_JSONL) as f:
            for line in f:
                e = json.loads(line)
                if str(e.get("participant_id")) != str(pid):
                    continue
                out = {}
                for k in PHQ8_KEYS:
                    if k in e:
                        s = e[k].get("score")
                        if isinstance(s, int):
                            out[k] = s
                        elif isinstance(s, str) and s.isdigit():
                            out[k] = int(s)
                        else:
                            out[k] = "N/A"
                return out
    except Exception as e:
        print(f"[WARN] baseline quant load failed: {e}")
    return None

def load_baseline_meta():
   
    if not os.path.exists(META_BASELINE_CSV):
        print(f"[WARN] baseline meta csv not found: {META_BASELINE_CSV}")
        return {}
    try:
        df = pd.read_csv(META_BASELINE_CSV)
        pid_col = next((c for c in df.columns if 'participant' in c.lower()), df.columns[0])
        sev_col = next((c for c in df.columns if 'severity' in c.lower()), None)
        out = {}
        if sev_col is not None:
            for _, r in df.iterrows():
                v = pd.to_numeric(r[sev_col], errors='coerce')
                if pd.notna(v):
                    out[str(int(r[pid_col]))] = int(v)
        else:
            text_cols = [c for c in df.columns if df[c].dtype == object]
            for _, r in df.iterrows():
                sev = None
                for c in text_cols:
                    m = re.search(r"<severity>\s*([0-4])", str(r[c]))
                    if m:
                        sev = int(m.group(1))
                        break
                if sev is not None:
                    out[str(int(r[pid_col]))] = sev
        print(f"Baseline meta severities loaded for {len(out)} participants "
              f"from {META_BASELINE_CSV} (columns: {list(df.columns)})")
        return out
    except Exception as e:
        print(f"[WARN] baseline meta load failed: {e}")
        return {}

def load_quan_xml_from_baseline(pid):
    
    if not os.path.exists(QUAN_BASELINE_JSONL):
        return None
    try:
        with open(QUAN_BASELINE_JSONL) as f:
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
                return ('\n'.join(out) + '\n') if out else None
    except Exception as e:
        print(f"[WARN] baseline quant xml load failed: {e}")
    return None

def load_transcript(pid):
    path = os.path.join(DATA_DIR, f"transcripts/{pid}_TRANSCRIPT.csv")
    if not os.path.exists(path):
        return None
    raw = pd.read_csv(path, sep='\t')
    raw['speaker'] = raw['speaker'].fillna('Unknown').astype(str)
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

def load_judge_scores():
   
    dims = ["specificity", "completeness", "coherence", "accuracy"]
    for path in FEEDBACK_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            cols = {}
            for d in dims:
                match = [c for c in df.columns if d in c.lower()]
                if match:
                    cols[d] = match[0]
            if not cols:
                print(f"[WARN] feedback file found ({path}) but no judge-score columns recognised: {list(df.columns)}")
                return {}, path
            pid_col = next((c for c in df.columns if 'participant' in c.lower()), None)
            if pid_col is None:
                print(f"[WARN] no participant id column in {path}")
                return {}, path
            df = df.sort_index().groupby(df[pid_col].astype(str), as_index=True).last()
            scores = {}
            for pid_str, row in df.iterrows():
                vals = []
                for d, c in cols.items():
                    v = pd.to_numeric(row[c], errors='coerce')
                    if pd.notna(v):
                        vals.append(float(v))
                if vals:
                    scores[pid_str] = sum(vals) / len(vals)
            print(f"Judge scores loaded from {path} (columns: {cols}) for {len(scores)} participants")
            return scores, path
        except Exception as e:
            print(f"[WARN] failed reading {path}: {e}")
    print("[WARN] no feedback/judge file found - C3 will use coverage + consensus only")
    return {}, None

QUANT_SYSTEM = """You are a AI psychiatrist assessment tool specialized in analyzing interview transcripts to evaluate PHQ-8 (Patient Health Questionnaire-8) scores. The PHQ-8 is a validated depression screening instrument that assesses eight of the nine DSM-IV criteria for depressive disorders over the past 2 weeks.

PHQ-8 Scoring System:
- 0 = Not at all (0-1 days in the past 2 weeks)
- 1 = Several days (2-6 days)
- 2 = More than half the days (7-11 days)
- 3 = Nearly every day (12-14 days)

Key Principles:
1. Base all assessments strictly on evidence from the transcript
2. When no relevant evidence exists for a symptom, output "N/A" rather than assuming absence (score of 0)
3. Consider both explicit statements and behavioral indicators in the interview
4. Be conservative - only assign scores when there is clear evidence of frequency
5. Remember that absence of mention does not necessarily mean absence of symptom
6. Follow the format specified exactly"""

def quant_prompt(transcript):
    return f"""Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

Interview transcript:
<transcript>
{transcript}
</transcript>

Analyze each symptom using the following approach in <thinking> tags:
1. Search for direct quotes or behavioral evidence related to each PHQ-8 symptom
2. Evaluate the frequency/severity based on available evidence
3. If no relevant evidence exists, mark as "N/A" rather than assuming absence
4. Only assign numeric scores (0-3) when evidence clearly indicates frequency

After your analysis, provide your final assessment in <answer> tags as a JSON object.

For each symptom, provide:
1. "evidence": exact quotes from transcript (use "No relevant evidence found" if not discussed)
2. "reason": explanation of scoring decision, including why N/A if applicable
3. "score": integer 0-3 based on evidence, or "N/A" if no relevant evidence

Return ONLY a JSON object in <answer> tags with these exact keys:
- "PHQ8_NoInterest": {{evidence, reason, score}} for little interest or pleasure in doing things (anhedonia)
- "PHQ8_Depressed": {{evidence, reason, score}} for feeling down, depressed, or hopeless (depressed mood)
- "PHQ8_Sleep": {{evidence, reason, score}} for trouble falling or staying asleep, or sleeping too much (sleep problems)
- "PHQ8_Tired": {{evidence, reason, score}} for feeling tired or having little energy (fatigue)
- "PHQ8_Appetite": {{evidence, reason, score}} for poor appetite or overeating (appetite/weight changes)
- "PHQ8_Failure": {{evidence, reason, score}} for feeling bad about yourself or that you are a failure (negative self-perception)
- "PHQ8_Concentrating": {{evidence, reason, score}} for trouble concentrating on things like reading or watching TV (concentration problems)
- "PHQ8_Moving": {{evidence, reason, score}} for moving or speaking slowly, or being fidgety/restless (psychomotor changes)"""

def chat(messages, options, logprobs=False, timeout=1800):
    payload = {"model": MODEL, "messages": messages, "stream": False, "options": options}
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 20
    r = requests.post(CHAT_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def token_view(data):
    """Returns (toks, full_text, char_to_tok) from a logprobs response."""
    lp = data.get("logprobs")
    if not lp:
        return None, "", []
    token_list = lp.get("content") or lp.get("tokens") or [] if isinstance(lp, dict) else lp
    toks, full, c2t = [], "", []
    for i, t in enumerate(token_list):
        tk = {"i": i, "token": t.get("token", ""), "logprob": t.get("logprob"),
              "top": t.get("top_logprobs", [])}
        toks.append(tk)
        for _ in tk["token"]:
            c2t.append(i)
        full += tk["token"]
    return toks, full, c2t

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
    if severity is None:
        return None
    return 1 if severity >= 2 else 0

# ─── STEP A: quant with logprobs + verified symptom mapping ──────────────────

def run_quant(pid, transcript):
    """Runs quant agent with logprobs.
    Returns dict with: scores {sym: int|'N/A'}, probs {sym: float},
    match {sym: bool}, na_rate, quant_xml (for meta prompt), or None."""
    print("\n--- STEP A: quant agent (temp 0, top_k 1) with logprobs ---")
    msgs = [{"role": "system", "content": QUANT_SYSTEM},
            {"role": "user", "content": quant_prompt(transcript)}]
    try:
        data = chat(msgs, {"temperature": 0, "top_k": 1, "top_p": 1.0}, logprobs=True)
    except Exception as e:
        print(f"[ERROR] quant call failed: {e}")
        return None

    content = data["message"]["content"]
    toks, full, c2t = token_view(data)
    if toks is None:
        print("[ERROR] no logprobs in quant response")
        return None

    ans = extract_tag(content, "answer")
    if ans is None:
        a, b = content.find("{"), content.rfind("}")
        ans = content[a:b + 1] if a != -1 and b != -1 else None
    parsed = None
    if ans:
        cleaned = re.sub(r"```json|```", "", ans).strip()
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            print(f"[WARN] JSON parse failed: {e}")

    scores, probs, match = {}, {}, {}
    # map each symptom key -> position in FULL text -> "score" -> digit token
    positions = []
    for k in PHQ8_KEYS:
        p = full.find(f'"{k}"')
        if p != -1:
            positions.append((p, k))
    positions.sort()
    for idx, (p, k) in enumerate(positions):
        seg_end = positions[idx + 1][0] if idx + 1 < len(positions) else len(full)
        sp = full.find('"score"', p, seg_end)
        if sp == -1:
            continue
        j = sp + len('"score"')
        digit_pos = -1
        while j < seg_end:
            ch = full[j]
            if ch in "0123":
                digit_pos = j
                break
            if ch == "N" and full[j:j + 3] in ('N/A', 'N/a'):
                break
            if ch == '"' and full[j:j + 4] == '"N/A':
                break
            if ch == '}':
                break
            j += 1
        if digit_pos == -1:
            scores[k] = "N/A"
            continue
        tk = toks[c2t[digit_pos]]
        tok_digit = tk["token"].strip()
        prob = math.exp(tk["logprob"]) if tk["logprob"] is not None else None
        scores[k] = int(tok_digit) if tok_digit.isdigit() else None
        probs[k] = prob
        if parsed and k in parsed:
            jscore = parsed[k].get("score")
            jint = jscore if isinstance(jscore, int) else (
                int(jscore) if isinstance(jscore, str) and jscore.isdigit() else None)
            match[k] = (jint == scores[k])
        else:
            match[k] = None

    scored = [k for k in PHQ8_KEYS if isinstance(scores.get(k), int)]
    na = [k for k in PHQ8_KEYS if scores.get(k) == "N/A" or k not in scores]
    na_rate = len(na) / len(PHQ8_KEYS)

    print(f"scored symptoms: {len(scored)} | N/A: {len(na)} (rate {na_rate:.2f})")
    for k in PHQ8_KEYS:
        if k in scored:
            mk = {True: "MATCH", False: "MISMATCH!", None: "no-json"}[match.get(k)]
            print(f"  {k:<22} score={scores[k]}  prob={probs[k]:.4f}  [{mk}]")
        else:
            print(f"  {k:<22} N/A")

    xml = []
    for k in scored:
        reason = ""
        if parsed and k in parsed:
            reason = parsed[k].get("reason", "")
        xml.append(f"<{k.lower()}_score>{scores[k]}</{k.lower()}_score>")
        xml.append(f"<{k.lower()}_explanation>{reason}</{k.lower()}_explanation>")
    return {"scores": scores, "probs": probs, "match": match,
            "na_rate": na_rate,
            "quant_xml": ('\n'.join(xml) + '\n') if xml else '',
            "scored": scored}

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

STOPS_FOLLOWUP = """Thank you. Now, considering the severity prediction you just made, reason carefully about your confidence in the following structured steps:
1. In <supporting_evidence> tags: list the specific evidence from the transcript and assessments that supports your severity prediction.
2. In <uncertainty_factors> tags: list what is missing, ambiguous, or conflicting (e.g., symptoms not discussed, contradictory signals, sparse transcript, disagreement between qualitative and quantitative assessments).
3. In <confidence> tags: given steps 1 and 2, provide a single integer from 0 to 100 representing how confident you are that your severity prediction is correct (0 = no confidence, 100 = completely certain).
Do NOT change or restate a new severity. Provide all tags in XML format, each on a new line."""

def locate_severity(data):
    toks, full, c2t = token_view(data)
    if toks is None:
        return None, None, {}
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
    tk = toks[c2t[digit_pos]]
    prob = math.exp(tk["logprob"]) if tk["logprob"] is not None else None
    dist = {}
    for entry in tk["top"]:
        tt = entry.get("token", "").strip()
        if tt in ("0", "1", "2", "3", "4"):
            dist[tt] = round(math.exp(entry["logprob"]), 4)
    return tk["token"].strip(), prob, dist

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gt = load_ground_truth()
    print(f"Ground truth loaded for {len(gt)} subjects")
    judge, judge_path = load_judge_scores()
    baseline_meta = load_baseline_meta()

    meta_rows, symptom_rows = [], []

    for pid, note in SUBJECTS:
        print("\n" + "#" * 70)
        print(f"# Participant {pid}  [{note}]")
        print("#" * 70)

        transcript = load_transcript(pid)
        if transcript is None:
            print(f"[SKIP] no transcript")
            continue
        qual = load_qual(pid) or "(no qualitative assessment available)"

        g = gt.get(pid, {})
        gt_total = g.get('total')
        gt_label = (1 if gt_total >= 10 else 0) if gt_total is not None else None
        print(f"transcript {len(transcript)} chars | GT total {gt_total} -> binary {gt_label}")

        q = run_quant(pid, transcript)
        if q is None:
            quant_xml, na_rate = "(no quantitative assessment available)", None
        else:
            quant_xml, na_rate = q["quant_xml"] or "(all symptoms N/A)", q["na_rate"]
            saved = load_saved_quant_scores(pid)
            if saved is None:
                print("[NOTE] baseline quant jsonl entry not found for cross-check")
            else:
                diffs = [f"{k}: baseline={saved.get(k, 'N/A')} now={q['scores'].get(k, 'N/A')}"
                         for k in PHQ8_KEYS
                         if q["scores"].get(k, "N/A") != saved.get(k, "N/A")]
                if diffs:
                    print(f"[DETERMINISM CHECK] {len(diffs)} score(s) DIFFER from saved baseline:")
                    for d in diffs:
                        print(f"    {d}")
                else:
                    print("[DETERMINISM CHECK] regenerated scores MATCH saved baseline exactly")
            for k in q["scored"]:
                gt_item = g.get('items', {}).get(k)
                pred = q["scores"][k]
                symptom_rows.append({
                    "pid": pid, "symptom": k, "pred_score": pred,
                    "gt_score": gt_item,
                    "exact_correct": (pred == gt_item) if gt_item is not None else None,
                    "within1_correct": (abs(pred - gt_item) <= 1) if gt_item is not None else None,
                    "token_prob": round(q["probs"][k], 4),
                    "token_json_match": q["match"].get(k),
                })

        baseline_xml = load_quan_xml_from_baseline(pid)
        if baseline_xml is not None:
            quant_xml = baseline_xml
            print("[META INPUT] using quant block from saved baseline jsonl")
        else:
            print("[META INPUT] baseline quant block unavailable - using regenerated block")

        prompt = base_meta_prompt(transcript, qual, quant_xml)
        base_msgs = [{"role": "system", "content": META_SYSTEM},
                     {"role": "user", "content": prompt}]

        print("\n--- STEP B: S1 meta logprob (temp 0) ---")
        s1_sev = s1_prob = None
        s1_content = ""
        try:
            data = chat(base_msgs, {"temperature": 0, "top_k": 20, "top_p": 1}, logprobs=True)
            s1_content = data["message"]["content"]
            s1_sev = parse_severity(s1_content)
            digit, s1_prob, dist = locate_severity(data)
            if s1_prob is not None:
                print(f"severity: {s1_sev} | token '{digit}' prob {s1_prob:.4f} | dist {dist}")
            else:
                print(f"severity: {s1_sev} | logprob locate FAILED")
        except Exception as e:
            print(f"[ERROR] S1 failed: {e}")

        s1_binary = to_binary(s1_sev)
        s1_correct = (s1_binary == gt_label) if (s1_binary is not None and gt_label is not None) else None

        # cross-check S1 prediction against the ORIGINAL baseline meta prediction
        base_sev = baseline_meta.get(str(pid))
        if base_sev is None:
            print("[BASELINE CHECK] no baseline severity found for this pid")
        elif s1_sev == base_sev:
            print(f"[BASELINE CHECK] S1 severity MATCHES original baseline ({base_sev})")
        else:
            print(f"[BASELINE CHECK] S1 severity {s1_sev} DIFFERS from original baseline {base_sev}")

        print("\n--- STEP B: S2 second-turn SToPS confidence ---")
        s2_conf = None
        if s1_content:
            try:
                follow_msgs = base_msgs + [
                    {"role": "assistant", "content": s1_content},
                    {"role": "user", "content": STOPS_FOLLOWUP},
                ]
                data = chat(follow_msgs, {"temperature": 0, "top_k": 20, "top_p": 1})
                content = data["message"]["content"]
                conf_raw = extract_tag(content, "confidence")
                m = re.search(r"\d+", conf_raw or "")
                s2_conf = min(100, max(0, int(m.group()))) if m else None
                unc = extract_tag(content, "uncertainty_factors")
                print(f"verbalized confidence: {s2_conf}")
                if unc:
                    print(f"uncertainty (first 250 chars): {unc[:250]}")
            except Exception as e:
                print(f"[ERROR] S2 failed: {e}")

        print(f"\n--- STEP B: S3 sampling ({N_SAMPLES} @ temp {SAMPLE_TEMP}) ---")
        sample_sevs = []
        for k in range(N_SAMPLES):
            try:
                data = chat(base_msgs, {"temperature": SAMPLE_TEMP, "top_k": 20,
                                        "top_p": 1, "seed": 1001 + k})
                sev = parse_severity(data["message"]["content"])
                sample_sevs.append(sev)
                print(f"  sample {k+1}: severity {sev}")
            except Exception as e:
                print(f"  sample {k+1}: [ERROR] {e}")
                sample_sevs.append(None)
        valid = [s for s in sample_sevs if s is not None]
        agree_bin = (sum(1 for s in valid if to_binary(s) == s1_binary) / len(valid)) \
            if valid and s1_binary is not None else None
        agree_exact = (sum(1 for s in valid if s == s1_sev) / len(valid)) \
            if valid and s1_sev is not None else None
        print(f"agreement: binary={agree_bin} exact={agree_exact}")

        print("\n--- STEP C: C3 pipeline-internal signals ---")
        comps = {}
        jraw = judge.get(str(pid))
        if jraw is not None:
            comps['judge_norm'] = jraw / 5.0
        if na_rate is not None:
            comps['coverage'] = 1.0 - na_rate
        if q and q["scored"] and s1_binary is not None:
            mean_score = sum(q["scores"][k] for k in q["scored"]) / len(q["scored"])
            consensus = (mean_score / 3.0) if s1_binary == 1 else (1.0 - mean_score / 3.0)
            comps['consensus'] = consensus
        c3 = round(sum(comps.values()) / len(comps), 3) if comps else None
        print(f"components: { {k: round(v,3) for k,v in comps.items()} }  ->  C3 = {c3}")

        meta_rows.append({
            "pid": pid, "note": note,
            "gt_total": gt_total, "gt_binary": gt_label,
            "s1_severity": s1_sev, "baseline_sev": base_sev,
            "s1_binary": s1_binary, "correct": s1_correct,
            "s1_prob": round(s1_prob, 4) if s1_prob else None,
            "s2_conf": s2_conf,
            "s3_agree_binary": agree_bin, "s3_agree_exact": agree_exact,
            "s3_samples": "|".join(str(s) for s in sample_sevs),
            "na_rate": na_rate, "c3": c3,
            "c3_components": ";".join(f"{k}={v:.3f}" for k, v in comps.items()),
        })

    print("\n\n" + "=" * 110)
    print("META-LEVEL SUMMARY (compare WRONG rows vs CORRECT rows per column)")
    print("=" * 110)
    print(f"{'PID':<5}{'Note':<24}{'GT':<4}{'Sev':<5}{'Base':<6}{'OK?':<7}{'S1prob':<9}{'S2conf':<8}{'S3agr':<7}{'C3':<7}{'NA%':<6}{'S3 samples':<13}")
    print("-" * 110)
    for r in meta_rows:
        print(f"{r['pid']:<5}{r['note']:<24}{str(r['gt_binary']):<4}{str(r['s1_severity']):<5}"
              f"{str(r['baseline_sev']):<6}"
              f"{str(r['correct']):<7}{str(r['s1_prob']):<9}{str(r['s2_conf']):<8}"
              f"{str(r['s3_agree_binary']):<7}{str(r['c3']):<7}"
              f"{str(round(r['na_rate'],2)) if r['na_rate'] is not None else 'None':<6}"
              f"{r['s3_samples']:<13}")
    print("-" * 110)

    sr = [r for r in symptom_rows if r["exact_correct"] is not None]
    if sr:
        print("\n" + "=" * 110)
        print("QUANT-LEVEL SUMMARY (per-symptom logprobs vs per-symptom ground truth)")
        print("=" * 110)
        n = len(sr)
        n_exact = sum(1 for r in sr if r["exact_correct"])
        n_w1 = sum(1 for r in sr if r["within1_correct"])
        mp_corr = [r["token_prob"] for r in sr if r["exact_correct"]]
        mp_wrong = [r["token_prob"] for r in sr if not r["exact_correct"]]
        mism = sum(1 for r in symptom_rows if r["token_json_match"] is False)
        print(f"scored symptom predictions with GT: {n}")
        print(f"exact-match accuracy : {n_exact}/{n} = {n_exact/n:.2f}")
        print(f"within-1 accuracy    : {n_w1}/{n} = {n_w1/n:.2f}")
        if mp_corr:
            print(f"mean token prob when EXACT-CORRECT : {sum(mp_corr)/len(mp_corr):.4f}  (n={len(mp_corr)})")
        if mp_wrong:
            print(f"mean token prob when WRONG         : {sum(mp_wrong)/len(mp_wrong):.4f}  (n={len(mp_wrong)})")
        print(f"token-vs-JSON cross-check mismatches : {mism} (should be 0)")
        print("\nKEY QUESTION: is mean prob on WRONG symptom scores lower than on CORRECT ones?")

    if meta_rows:
        p1 = os.path.join(OUT_DIR, "pretest_full_meta.csv")
        with open(p1, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            w.writeheader(); w.writerows(meta_rows)
        print(f"\nSaved: {p1}")
    if symptom_rows:
        p2 = os.path.join(OUT_DIR, "pretest_full_symptoms.csv")
        with open(p2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(symptom_rows[0].keys()))
            w.writeheader(); w.writerows(symptom_rows)
        print(f"Saved: {p2}")

if __name__ == "__main__":
    main()
