

import os
import re
import json
import math
import requests
import pandas as pd

OLLAMA_NODE = os.environ.get("OLLAMA_NODE", "localhost")
CHAT_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
MODEL = "gemma3:27b"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.environ.get("ROOTDIR", os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
DATA_DIR = os.environ.get("DAIC_WOZ_DATA", os.path.abspath(os.path.join(ROOTDIR, "..", "daic_woz_data")))
QUAL_PATH = os.path.join(ROOTDIR, "analysis_output", "Baseline", "qual", "qual_assessment_GEMMA_v2.csv")
FEEDBACK_PATH = os.path.join(ROOTDIR, "analysis_output", "Baseline", "qual", "feedback_evaluations_v2.csv")
QUAN_BASELINE_JSONL = os.path.join(ROOTDIR, "analysis_output", "Baseline", "quan", "results_zero_shot_test41_detailed.jsonl")
META_BASELINE_CSV = os.path.join(ROOTDIR, "analysis_output", "Baseline", "qual", "meta_review_zeroshot_test_v2.csv")
OUT_DIR = os.path.join(ROOTDIR, "analysis_output", "rq3")

PHQ8_KEYS = [
    'PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired',
    'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving'
]

QUANT_OPTIONS = {"temperature": 0, "top_k": 1, "top_p": 1.0}
META_OPTIONS = {"temperature": 0, "top_k": 20, "top_p": 1}
SAMPLE_TEMP = 0.7
SEED_BASE = 1001

def load_ground_truth():
    """{pid: {'total': float, 'items': {PHQ8_*: float}}} from train+dev AVEC labels."""
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
            binary = int(row['PHQ8_Binary']) if 'PHQ8_Binary' in df.columns and pd.notna(row.get('PHQ8_Binary')) else None
            gt[pid] = {'total': total, 'binary': binary, 'items': items}
    return gt

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
    if not os.path.exists(FEEDBACK_PATH):
        print(f"[WARN] feedback file not found: {FEEDBACK_PATH}")
        return {}
    try:
        df = pd.read_csv(FEEDBACK_PATH)
        cols = {d: next(c for c in df.columns if d in c.lower()) for d in dims
                if any(d in c.lower() for c in df.columns)}
        pid_col = next((c for c in df.columns if 'participant' in c.lower()), None)
        if not cols or pid_col is None:
            print(f"[WARN] judge columns not recognised in {FEEDBACK_PATH}")
            return {}
        df = df.sort_index().groupby(df[pid_col].astype(str), as_index=True).last()
        scores = {}
        for pid_str, row in df.iterrows():
            vals = [float(v) for v in (pd.to_numeric(row[c], errors='coerce') for c in cols.values())
                    if pd.notna(v)]
            if vals:
                scores[pid_str] = sum(vals) / len(vals)
        print(f"Judge scores loaded for {len(scores)} participants")
        return scores
    except Exception as e:
        print(f"[WARN] judge load failed: {e}")
        return {}

def load_baseline_meta():
    
    if not os.path.exists(META_BASELINE_CSV):
        print(f"[WARN] baseline meta csv not found: {META_BASELINE_CSV}")
        return {}
    df = pd.read_csv(META_BASELINE_CSV)
    pid_col = next((c for c in df.columns if 'participant' in c.lower()), df.columns[0])
    sev_col = next((c for c in df.columns if 'severity' in c.lower()), None)
    out = {}
    if sev_col is not None:
        for _, r in df.iterrows():
            v = pd.to_numeric(r[sev_col], errors='coerce')
            if pd.notna(v):
                out[str(int(r[pid_col]))] = int(v)
    print(f"Baseline meta severities loaded for {len(out)} participants")
    return out

def baseline_pid_list():
   
    df = pd.read_csv(META_BASELINE_CSV)
    pid_col = next((c for c in df.columns if 'participant' in c.lower()), df.columns[0])
    return [int(p) for p in df[pid_col].tolist()]

def load_saved_quant_scores(pid):
    
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

def chat(messages, options, logprobs=False, timeout=1800):
    payload = {"model": MODEL, "messages": messages, "stream": False, "options": options}
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 20
    r = requests.post(CHAT_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def token_view(data):
  
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

def locate_severity(data):
    
    toks, full, c2t = token_view(data)
    if toks is None:
        return None, None, {}, None
    idx = full.find("<severity>")
    if idx == -1:
        return None, None, {}, None
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
        return None, None, {}, None
    tk = toks[c2t[digit_pos]]
    prob = math.exp(tk["logprob"]) if tk["logprob"] is not None else None
    dist = {}
    for entry in tk["top"]:
        tt = entry.get("token", "").strip()
        if tt in ("0", "1", "2", "3", "4"):
            dist[tt] = round(math.exp(entry["logprob"]), 6)
    return tk["token"].strip(), prob, dist, tk["i"]

def p_depressed_from_dist(dist):
   
    if not dist:
        return None
    tot = sum(dist.values())
    if tot <= 0:
        return None
    return sum(dist.get(d, 0.0) for d in ("2", "3", "4")) / tot

def parse_diagnosis(content):
  
    d = content.find("<diagnosis>")
    s = content.find("<severity>")
    if d == -1:
        return False, None
    return (s == -1 or d < s), extract_tag(content, "diagnosis")

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

QUANT_S2_FOLLOWUP = """Thank you. Now, considering the PHQ-8 scores you just assigned, reason about your confidence in each one. Do NOT change any scores.
For each symptom you assigned a numeric score (skip the N/A ones), provide a single integer from 0 to 100 representing how confident you are that the score is correct (0 = no confidence, 100 = completely certain).
Return ONLY a JSON object in <answer> tags mapping each scored symptom key to its confidence integer, for example:
<answer>
{"PHQ8_Depressed": 80, "PHQ8_Sleep": 65}
</answer>"""

def run_quant(pid, transcript):
    
    msgs = [{"role": "system", "content": QUANT_SYSTEM},
            {"role": "user", "content": quant_prompt(transcript)}]
    data = chat(msgs, QUANT_OPTIONS, logprobs=True)
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
            print(f"[WARN] quant JSON parse failed: {e}")

    scores, probs, dists, match = {}, {}, {}, {}
    positions = sorted((full.find(f'"{k}"'), k) for k in PHQ8_KEYS if full.find(f'"{k}"') != -1)
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
        scores[k] = int(tok_digit) if tok_digit.isdigit() else None
        probs[k] = math.exp(tk["logprob"]) if tk["logprob"] is not None else None
        d = {}
        for entry in tk["top"]:
            tt = entry.get("token", "").strip()
            if tt in ("0", "1", "2", "3"):
                d[tt] = round(math.exp(entry["logprob"]), 6)
        dists[k] = d
        if parsed and k in parsed:
            jscore = parsed[k].get("score")
            jint = jscore if isinstance(jscore, int) else (
                int(jscore) if isinstance(jscore, str) and jscore.isdigit() else None)
            match[k] = (jint == scores[k])
        else:
            match[k] = None

    scored = [k for k in PHQ8_KEYS if isinstance(scores.get(k), int)]
    na = [k for k in PHQ8_KEYS if scores.get(k) == "N/A" or k not in scores]
    return {"scores": scores, "probs": probs, "dists": dists, "match": match,
            "na_rate": len(na) / len(PHQ8_KEYS), "content": content, "scored": scored}

def run_quant_s2(quant_messages_content, transcript):
   
    msgs = [{"role": "system", "content": QUANT_SYSTEM},
            {"role": "user", "content": quant_prompt(transcript)},
            {"role": "assistant", "content": quant_messages_content},
            {"role": "user", "content": QUANT_S2_FOLLOWUP}]
    data = chat(msgs, QUANT_OPTIONS)
    content = data["message"]["content"]
    ans = extract_tag(content, "answer")
    if ans is None:
        a, b = content.find("{"), content.rfind("}")
        ans = content[a:b + 1] if a != -1 and b != -1 else None
    try:
        cleaned = re.sub(r"```json|```", "", ans).strip()
        parsed = json.loads(cleaned)
        out = {}
        for k, v in parsed.items():
            if k in PHQ8_KEYS:
                m = re.search(r"\d+", str(v))
                if m:
                    out[k] = min(100, max(0, int(m.group())))
        return out
    except Exception as e:
        print(f"[WARN] quant S2 parse failed: {e} | head: {content[:150]!r}")
        return {}

META_SYSTEM = ("You are an AI psychiatrist assistant specializing in depression. "
               "Your task is to analyze the interview transcript and related qualitative "
               "and quantitative assessments, and then predict the diagnosis label and severity level.")

QUAL_PLACEHOLDER = "(no qualitative assessment available)"
QUAN_PLACEHOLDER = "(no quantitative assessment available)"

_ENDING_ANSWER_FIRST = """Please predict their severity level in <severity> tags, where 0 means no significant depressive symptoms, 1 means mild symptoms, 2 means moderate symptoms, 3 means moderately severe symptoms, and 4 means severe symptoms. \
        Please explain your predictions in <explanation> tags. Please provide answers in the XML format with each tag on a new line.
        """

# DECLARED DEVIATION (explanation_first condition only): explanation requested
_ENDING_EXPLANATION_FIRST = """Please first explain your reasoning in <explanation> tags. Then, after the explanation, predict their severity level in <severity> tags, where 0 means no significant depressive symptoms, 1 means mild symptoms, 2 means moderate symptoms, 3 means moderately severe symptoms, and 4 means severe symptoms. \
        Please provide answers in the XML format with each tag on a new line.
        """

def build_meta_prompt(transcript, qual, quan, order="answer_first"):
    ending = _ENDING_ANSWER_FIRST if order == "answer_first" else _ENDING_EXPLANATION_FIRST
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
        {ending}"""

STOPS_FOLLOWUP = """Thank you. Now, considering the severity prediction you just made, reason carefully about your confidence in the following structured steps:
1. In <supporting_evidence> tags: list the specific evidence from the transcript and assessments that supports your severity prediction.
2. In <uncertainty_factors> tags: list what is missing, ambiguous, or conflicting (e.g., symptoms not discussed, contradictory signals, sparse transcript, disagreement between qualitative and quantitative assessments).
3. In <confidence> tags: given steps 1 and 2, provide a single integer from 0 to 100 representing how confident you are that your severity prediction is correct (0 = no confidence, 100 = completely certain).
Do NOT change or restate a new severity. Provide all tags in XML format, each on a new line."""

def meta_inputs_for_condition(condition, transcript, qual, quan_xml):
   
    qual_text = qual if qual is not None else QUAL_PLACEHOLDER
    quan_text = quan_xml if quan_xml is not None else QUAN_PLACEHOLDER
    if condition == "main":
        return qual_text, quan_text, "answer_first"
    if condition == "no_qual":
        return QUAL_PLACEHOLDER, quan_text, "answer_first"
    if condition == "no_quant":
        return qual_text, QUAN_PLACEHOLDER, "answer_first"
    if condition == "transcript_only":
        return QUAL_PLACEHOLDER, QUAN_PLACEHOLDER, "answer_first"
    if condition == "explanation_first":
        return qual_text, quan_text, "explanation_first"
    raise ValueError(f"unknown condition: {condition}")
