

import os
import json
import math
import requests
import pandas as pd

OLLAMA_NODE = os.environ.get("OLLAMA_NODE", "localhost")
GENERATE_URL = f"http://{OLLAMA_NODE}:11434/api/generate"
MODEL = "gemma3:27b"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.environ.get("ROOTDIR", os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
DATA_DIR = os.environ.get("DAIC_WOZ_DATA", os.path.abspath(os.path.join(ROOTDIR, "..", "daic_woz_data")))

TEST_SUBJECTS = [316, 319, 330, 339, 345]

SEVERITY_TOKENS = ["0", "1", "2", "3", "4"]

def load_transcript(pid):
    path = os.path.join(DATA_DIR, f"transcripts/{pid}_TRANSCRIPT.csv")
    if not os.path.exists(path):
        return None
    raw = pd.read_csv(path, sep='\t')
    raw['speaker'] = raw['speaker'].fillna('Unknown').astype(str)
    raw['value'] = raw['value'].fillna('').astype(str)
    return '\n'.join(raw['speaker'] + ': ' + raw['value'])

def check1_logprobs_available():
    """CHECK 1: Confirm Ollama returns a logprobs field at all."""
    print("\n" + "="*60)
    print("CHECK 1: Are logprobs returned by Ollama for gemma3:27b?")
    print("="*60)

    payload = {
        "model": MODEL,
        "prompt": "The capital of France is",
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "options": {"temperature": 0, "num_predict": 1}
    }

    try:
        r = requests.post(GENERATE_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [FAIL] Request errored: {e}")
        return False

    if "logprobs" not in data:
        print("  [FAIL] No 'logprobs' field in response.")
        print("  Response keys:", list(data.keys()))
        print("  -> C4 is NOT available with this setup.")
        return False

    print("  [PASS] 'logprobs' field present.")
    print("  Sample logprobs structure:")
    print(json.dumps(data["logprobs"][:1], indent=2))
    return True

def check2_mcsb_ability():
    """
    CHECK 2: When forced to output a single severity digit, does the model
    distribute probability sensibly over {0,1,2,3,4}?
    """
    print("\n" + "="*60)
    print("CHECK 2: Does Gemma 3 27B bind severity digits sensibly (MCSB)?")
    print("="*60)

    for pid in TEST_SUBJECTS:
        transcript = load_transcript(pid)
        if transcript is None:
            print(f"\n  [SKIP] No transcript for {pid}")
            continue

        if len(transcript) > 12000:
            transcript = transcript[:12000]

        prompt = f"""You are an AI psychiatrist assistant. Based on the interview transcript below, predict the participant's depression severity level.

<transcript>
{transcript}
</transcript>

Severity scale: 0=no significant symptoms, 1=mild, 2=moderate, 3=moderately severe, 4=severe.

Output ONLY a single digit (0, 1, 2, 3, or 4) for the severity level. Do not output anything else.

Severity:"""

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "options": {"temperature": 0, "num_predict": 1}
        }

        try:
            r = requests.post(GENERATE_URL, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"\n  [ERROR] {pid}: {e}")
            continue

        if "logprobs" not in data or not data["logprobs"]:
            print(f"\n  [ERROR] {pid}: no logprobs returned")
            continue

        first_token = data["logprobs"][0]
        chosen_token = first_token.get("token", "?")
        top = first_token.get("top_logprobs", [])

        token_probs = {}
        for entry in top:
            tok = entry["token"].strip()
            token_probs[tok] = math.exp(entry["logprob"])

        severity_mass = {d: token_probs.get(d, 0.0) for d in SEVERITY_TOKENS}
        total_severity_mass = sum(severity_mass.values())

        print(f"\n  Participant {pid}:")
        print(f"    Model output token: '{chosen_token}'")
        print(f"    Prob over severity digits:")
        for d in SEVERITY_TOKENS:
            bar = "#" * int(severity_mass[d] * 40)
            print(f"      {d}: {severity_mass[d]:.3f} {bar}")
        print(f"    Total mass on valid digits {{0-4}}: {total_severity_mass:.3f}")

        if total_severity_mass < 0.5:
            print("    [WARN] Less than half the mass on valid digits.")
            print("           Model may not be binding digits to severity. MCSB risk.")
        elif chosen_token not in SEVERITY_TOKENS:
            print("    [WARN] Chosen token is not a severity digit.")
        else:
            print("    [OK] Mass concentrated on valid severity digits.")

    print("\n" + "-"*60)
    print("INTERPRETATION GUIDE:")
    print("  GOOD: across subjects, the digit with highest prob VARIES with")
    print("        how depressed the subject seems, and >0.5 mass is on {0-4}.")
    print("  BAD:  the same digit always wins regardless of subject, OR most")
    print("        mass is on non-digit tokens. -> C4 unreliable, reconsider.")
    print("-"*60)

if __name__ == "__main__":
    print("C4 VERIFICATION TEST")
    print(f"Model: {MODEL}  Node: {OLLAMA_NODE}")

    logprobs_ok = check1_logprobs_available()

    if not logprobs_ok:
        print("\nCheck 1 failed - skipping Check 2. C4 not available.")
    else:
        check2_mcsb_ability()

    print("\nDone. Read the output above before trusting C4.")
