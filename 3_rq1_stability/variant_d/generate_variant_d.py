

import os
import re
import json
import random
import hashlib
import requests
import pandas as pd
import time
import argparse
from collections import defaultdict

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

parser = argparse.ArgumentParser()
parser.add_argument('--rate', type=int, required=True, choices=[5, 10, 20, 50],
                    help='Replacement rate as integer percentage')
args = parser.parse_args()
RATE_PERCENT = args.rate
RATE = RATE_PERCENT / 100.0

OLLAMA_NODE  = os.environ.get("OLLAMA_NODE", "localhost")
BASE_URL     = f"http://{OLLAMA_NODE}:11434/api/chat"
MODEL        = "gemma3:27b"

ORIGINAL_DIR = f"{DAIC}/transcripts"
OUTPUT_BASE  = f"{ROOT}/rq1_perturbations/variant_d/rate_{RATE_PERCENT}"

SEEDS        = [1, 2, 3, 4, 5]
TEMPERATURE  = 0.3
MAX_RETRIES  = 3
TIMEOUT      = 60

TEST_IDS = [
    316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383,
    385, 386, 389, 390, 393, 409, 413, 417, 422, 423, 427, 428, 430,
    436, 441, 445, 447, 449, 451, 455, 456, 459, 468, 472, 484, 485,
    487, 489
]

SPEAKER_LABELS = {"participant", "ellie"}

PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves"
}


NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion", "dozen",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth"
}

TIME_UNITS = {
    "year", "years", "month", "months", "week", "weeks",
    "day", "days", "hour", "hours", "minute", "minutes",
    "decade", "decades", "century", "centuries"
}

NEGATIONS = {"not", "no"}

EXCLUDE_WORDS = SPEAKER_LABELS | PRONOUNS | NUMBER_WORDS | TIME_UNITS | NEGATIONS

def get_transcript_rng(seed, rate_percent, participant_id):
    """
    One hash-seeded RNG per (seed, rate, participant).
    Used for global word selection across the whole transcript.
    Fully deterministic and reproducible.
    """
    s = f"{seed}_{rate_percent}_{participant_id}"
    h = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)
    return random.Random(h)

def tokenize(text):
    """
    Split into alternating (token, is_word) pairs, preserving all
    spacing and punctuation so the text reconstructs exactly.
    """
    segments = re.split(r'(\b[a-zA-Z]+\b)', text)
    tokens = []
    for seg in segments:
        if re.match(r'^[a-zA-Z]+$', seg):
            tokens.append((seg, True))
        else:
            tokens.append((seg, False))
    return tokens

def get_eligible_indices(tokens):
    """
    Return token indices eligible for synonym replacement.
    Excludes:
      - non-word tokens
      - EXCLUDE_WORDS (speaker labels, pronouns, number words,
        time units, negations)
      - ALL words inside vocal annotation tags <...>  — FIX 4: state machine
        handles multi-word tags like <clears throat> correctly
      - tokens adjacent to an apostrophe (contraction parts)
    """
    eligible = []
    in_annotation = False

    for i, (tok, is_word) in enumerate(tokens):
        if not is_word:
            if tok.strip() == '<':
                in_annotation = True
            elif tok.strip() == '>':
                in_annotation = False
            continue

        if in_annotation:
            continue

        if tok.lower() in EXCLUDE_WORDS:
            continue

        prev_tok = tokens[i-1][0] if i > 0 else ''
        next_tok = tokens[i+1][0] if i < len(tokens)-1 else ''

        if "'" in prev_tok or "'" in next_tok:
            continue

        eligible.append(i)
    return eligible

def get_synonyms_from_gemma(sentence, words_to_replace):
    """
    Ask Gemma for context-aware single-word synonyms.
    Returns {original_word: synonym} or {} on failure.
    """
    unique_words = list(dict.fromkeys(
        [w for w in words_to_replace if w.lower() not in EXCLUDE_WORDS]
    ))
    if not unique_words:
        return {}

    prompt = (
        f"For each word listed below, provide ONE synonym that fits "
        f"grammatically in the sentence. Use the sentence context to "
        f"choose the correct grammatical form (correct tense, plural, etc).\n\n"
        f"Sentence: \"{sentence}\"\n"
        f"Words to replace: {json.dumps(unique_words)}\n\n"
        f"Rules:\n"
        f"- Return ONLY a valid JSON object\n"
        f"- Each key is the original word, each value is its synonym\n"
        f"- The synonym must fit naturally in the sentence\n"
        f"- Each synonym must be exactly ONE single word — no phrases, "
        f"no hyphens, no multi-word expressions\n"
        f"- Do not explain anything\n"
        f"- Do not use markdown\n\n"
        f"Example: {{\"tired\": \"exhausted\", \"sad\": \"unhappy\"}}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                BASE_URL,
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": TEMPERATURE, "top_k": 20, "top_p": 0.9}
                },
                timeout=TIMEOUT
            )
            if response.status_code != 200:
                print(f"      [attempt {attempt}] HTTP {response.status_code}")
                time.sleep(3)
                continue

            content = response.json()['message']['content'].strip()
            if '```' in content:
                content = re.sub(r'```json|```', '', content).strip()

            match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            else:
                print(f"      [attempt {attempt}] No JSON found")
                time.sleep(3)

        except json.JSONDecodeError as e:
            print(f"      [attempt {attempt}] JSON error: {e}")
            time.sleep(3)
        except Exception as e:
            print(f"      [attempt {attempt}] Error: {e}")
            time.sleep(3)

    return {}

def apply_substitution(tokens, selected_indices, synonym_dict):
    """
    Substitute selected words using synonym_dict.
    Rejects: multi-word, hyphenated, non-alpha, number-word, and
    negation synonyms. Preserves original capitalisation.
    """
    new_tokens = list(tokens)

    for idx in selected_indices:
        original_word = tokens[idx][0]
        synonym = synonym_dict.get(original_word)
        if synonym is None:
            synonym = synonym_dict.get(original_word.lower())
        if synonym is None:
            continue

        synonym = synonym.strip()

        if ' ' in synonym or '-' in synonym:
            continue
        if not synonym or not synonym.isalpha():
            continue
        if synonym.lower() in NUMBER_WORDS or synonym.lower() in NEGATIONS:
            continue

        if original_word[0].isupper():
            synonym = synonym[0].upper() + synonym[1:]
        else:
            synonym = synonym.lower()

        new_tokens[idx] = (synonym, True)

    return ''.join(tok for tok, _ in new_tokens)

def process_transcript(df, rate, seed, participant_id):
    """
    FIX 2: global pool selection.

    Pass 1 — collect all eligible (row_idx, tok_idx) across whole transcript.
    Pass 2 — sample exactly round(N × rate) from the full pool at once.
    Pass 3 — group selected positions by turn.
    Pass 4 — call Gemma once per affected turn, apply synonyms.
    """
    modified_df   = df.copy()
    turns_skipped = 0

    all_eligible = []   
    turn_data    = {}   

    for row_idx, row in modified_df.iterrows():
        if str(row['speaker']).strip().lower() != 'participant':
            continue
        original_text = str(row['value']).strip() if pd.notna(row['value']) else ''
        if len(original_text.split()) < 3:
            turns_skipped += 1
            continue
        tokens   = tokenize(original_text)
        eligible = get_eligible_indices(tokens)
        if not eligible:
            turns_skipped += 1
            continue
        turn_data[row_idx] = (original_text, tokens)
        for tok_idx in eligible:
            all_eligible.append((row_idx, tok_idx))

    n_total = len(all_eligible)
    if n_total == 0:
        return modified_df, {
            'participant_id': participant_id, 'rate': RATE_PERCENT, 'seed': seed,
            'total_eligible': 0, 'total_selected': 0, 'selected_pct': 0.0,
            'turns_modified': 0, 'turns_failed': 0, 'turns_skipped': turns_skipped
        }

    n_select = round(n_total * rate)
    n_select = max(0, min(n_select, n_total))

    rng      = get_transcript_rng(seed, RATE_PERCENT, participant_id)
    selected = rng.sample(all_eligible, n_select) if n_select > 0 else []

    selected_by_turn = defaultdict(list)
    for (row_idx, tok_idx) in selected:
        selected_by_turn[row_idx].append(tok_idx)

    turns_modified = 0
    turns_failed   = 0

    for row_idx, tok_indices in selected_by_turn.items():
        original_text, tokens = turn_data[row_idx]
        words_to_replace      = [tokens[i][0] for i in tok_indices]
        synonym_dict          = get_synonyms_from_gemma(original_text, words_to_replace)
        if not synonym_dict:
            turns_failed += 1
            continue
        new_text = apply_substitution(tokens, tok_indices, synonym_dict)
        modified_df.at[row_idx, 'value'] = new_text
        turns_modified += 1

    return modified_df, {
        'participant_id': participant_id,
        'rate':           RATE_PERCENT,
        'seed':           seed,
        'total_eligible': n_total,
        'total_selected': n_select,
        'selected_pct':   round(n_select / n_total * 100, 1),
        'turns_modified': turns_modified,
        'turns_failed':   turns_failed,
        'turns_skipped':  turns_skipped
    }

def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    all_logs = []

    print(f"\n{'='*60}")
    print(f"Variant D — General Synonym Replacement (PRODUCTION)")
    print(f"Rate: {RATE_PERCENT}%  |  Seeds: {SEEDS}  |  Subjects: {len(TEST_IDS)}")
    print(f"Excludes: pronouns, numbers, time units, negations, annotations, contractions")
    print(f"Output: {OUTPUT_BASE}")
    print(f"{'='*60}")

    for seed in SEEDS:
        out_dir = os.path.join(OUTPUT_BASE, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n--- Seed {seed} ---")

        for participant_id in TEST_IDS:
            out_path  = os.path.join(out_dir, f"{participant_id}_TRANSCRIPT.csv")
            orig_path = os.path.join(ORIGINAL_DIR, f"{participant_id}_TRANSCRIPT.csv")

            if os.path.exists(out_path):
                print(f"  [{participant_id}] already exists, skipping")
                continue
            if not os.path.exists(orig_path):
                print(f"  [{participant_id}] original not found, skipping")
                continue

            print(f"  [{participant_id}] processing...", end=' ', flush=True)
            start = time.time()

            df = pd.read_csv(orig_path, sep='\t')
            df['speaker'] = df['speaker'].fillna('Unknown').astype(str)
            df['value']   = df['value'].fillna('').astype(str)

            modified_df, log = process_transcript(df, RATE, seed, participant_id)
            modified_df.to_csv(out_path, sep='\t', index=False)

            elapsed = round(time.time() - start, 1)
            print(f"done in {elapsed}s  "
                  f"(pool={log['total_eligible']}, "
                  f"selected={log['total_selected']}, "
                  f"pct={log['selected_pct']}%, "
                  f"turns_mod={log['turns_modified']}, "
                  f"failed={log['turns_failed']})")

            all_logs.append(log)

    log_path = os.path.join(OUTPUT_BASE, "generation_log.csv")
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        log_df = pd.concat([existing, pd.DataFrame(all_logs)], ignore_index=True)
    else:
        log_df = pd.DataFrame(all_logs)
    log_df.to_csv(log_path, index=False)

    print(f"\nGeneration log: {log_path}")
    print(f"Entries this run: {len(all_logs)}")

    if not log_df.empty:
        print("\n--- Summary ---")
        summary = log_df.groupby('seed').agg(
            mean_eligible=('total_eligible', 'mean'),
            mean_selected_pct=('selected_pct', 'mean'),
            mean_failed=('turns_failed', 'mean'),
            participants=('participant_id', 'count')
        ).round(2)
        print(summary.to_string())

    print(f"\nDone. Rate {RATE_PERCENT}% complete.")

if __name__ == "__main__":
    main()
