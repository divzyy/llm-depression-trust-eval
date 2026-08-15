
import os
import re
import time
import argparse
import requests
import pandas as pd
from datetime import datetime

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

ALL_TEST_IDS = [
    316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383, 385, 386, 389, 390, 393, 409, 413, 417, 422, 423, 427, 428, 430, 436, 441, 445, 447, 449, 451, 455, 456, 459, 468, 472, 484, 485, 487, 489

]

PARAPHRASE_SYSTEM_PROMPT = (
    "You are a precise paraphrasing assistant. "
    "Your only job is to rewrite the given text in different words while preserving its exact meaning. "
    "Do not add, remove, or change any information. "
    "Do not correct grammar unless it changes the meaning. "
    "Do not add any commentary, explanation, or prefix. "
    "Return only the rewritten text and nothing else."
)

PARAPHRASE_USER_TEMPLATE = (
    "Rewrite the following in different words. Keep the exact same meaning.\n\n"
    "Original: {utterance}\n\n"
    "Rewritten:"
)


_MARKER_PATTERN = re.compile(r'<[^>]+>|[A-Z][A-Z_]*_\d+')

def extract_markers(text: str):
    """
    Find all protected tokens (angle-bracket markers and WORD_N tokens)
    and replace with safe placeholders MRKR_0, MRKR_1, ...

    Returns (cleaned_text, marker_map) where marker_map maps
    placeholder -> original token.

    Examples:
        "<synch> yeah i'm <laughter> fine"
        -> "MRKR_0 yeah i'm MRKR_1 fine"
        -> {"MRKR_0": "<synch>", "MRKR_1": "<laughter>"}

        "INDICATOR_0 likely the city"
        -> "MRKR_0 likely the city"
        -> {"MRKR_0": "INDICATOR_0"}
    """
    marker_map = {}
    tokens = _MARKER_PATTERN.findall(text)
    seen = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
    token_to_placeholder = {}
    for i, token in enumerate(seen):
        placeholder = f"MRKR_{i}"
        marker_map[placeholder] = token
        token_to_placeholder[token] = placeholder
    cleaned = _MARKER_PATTERN.sub(lambda m: token_to_placeholder[m.group()], text)
    return cleaned, marker_map

def restore_markers(text: str, marker_map: dict) -> str:
    """
    Replace MRKR_N placeholders back with the original tokens.
    If Gemma dropped a placeholder, append the original token at the end
    so no annotation is silently lost.
    """
    restored = text
    missing = []
    for placeholder, original in marker_map.items():
        if placeholder in restored:
            restored = restored.replace(placeholder, original)
        else:
            missing.append(original)
    if missing:
        restored = restored.rstrip() + ' ' + ' '.join(missing)
    return restored

def paraphrase_utterance(utterance: str, participant_id: int,
                         base_url: str, model: str,
                         retries: int = 3) -> str:
    """
   

    Parameters
    ----------
    utterance      : original participant turn text
    participant_id : used only for logging
    base_url       : Ollama API endpoint
    model          : Ollama model name
    retries        : number of retry attempts on failure

    Returns
    -------
    str : paraphrased text, or original utterance if all retries fail
    """
    utterance = utterance.strip()

    #   - Disfluency fragments: "took um", "well like", "<laughter> probably um"
    if len(utterance) <= 15:
        return utterance

    cleaned_utterance, marker_map = extract_markers(utterance)

    prompt = PARAPHRASE_USER_TEMPLATE.format(utterance=cleaned_utterance)

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                base_url,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_k": 40,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )

            if response.status_code != 200:
                print(f"  [Attempt {attempt}] API error {response.status_code} "
                      f"for participant {participant_id}")
                time.sleep(5)
                continue

            content = response.json()["message"]["content"].strip()

            if not content:
                print(f"  [Attempt {attempt}] Empty response for participant "
                      f"{participant_id}, utterance: {utterance[:60]}")
                time.sleep(5)
                continue

            for prefix in ["Rewritten:", "Paraphrased:", "Output:", "Answer:"]:
                if content.startswith(prefix):
                    content = content[len(prefix):].strip()

            if marker_map:
                content = restore_markers(content, marker_map)

            return content

        except requests.exceptions.Timeout:
            print(f"  [Attempt {attempt}] Timeout for participant {participant_id}")
            time.sleep(10)
        except Exception as e:
            print(f"  [Attempt {attempt}] Exception: {e}")
            time.sleep(5)

    print(f"  WARNING: All retries failed for participant {participant_id}, "
          f"keeping original utterance.")
    return utterance

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Variant B: LLM-based paraphrase using Gemma 3 27B via Ollama. "
            "Only participant turns are modified. ELLIE turns are untouched."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=f"{DAIC}/transcripts",
        help="Directory containing original transcript CSV files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{ROOT}/rq1_perturbations/variant_b",
        help="Directory to write variant B transcript CSV files"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.expanduser("~/logs/variant_b"),
        help="Directory to write summary CSV and detailed log"
    )
    parser.add_argument(
        "--test_ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional: run on specific participant IDs only (for testing). "
            "Example: --test_ids 436 370 428. "
            "If not provided, runs on all 41 test subjects."
        )
    )
    args = parser.parse_args()

    ollama_node = os.environ.get("OLLAMA_NODE", "localhost")
    base_url    = f"http://{ollama_node}:11434/api/chat"
    model       = "gemma3:27b"

    print(f"DEBUG: OLLAMA_NODE = {ollama_node}")
    print(f"DEBUG: BASE_URL    = {base_url}")
    print(f"DEBUG: MODEL       = {model}")

    if args.test_ids is not None:
        invalid = [i for i in args.test_ids if i not in ALL_TEST_IDS]
        if invalid:
            print(f"WARNING: These IDs are not in the 41 test subjects "
                  f"and will be skipped: {invalid}")
        subject_ids = [i for i in args.test_ids if i in ALL_TEST_IDS]
        print(f"TEST MODE: running on {len(subject_ids)} subject(s): {subject_ids}")
    else:
        subject_ids = ALL_TEST_IDS
        print(f"FULL MODE: running on all {len(subject_ids)} test subjects")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir,    exist_ok=True)

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path    = os.path.join(args.log_dir, f"variant_b_summary_{timestamp}.csv")
    detail_log_path = os.path.join(args.log_dir, f"variant_b_detailed_log_{timestamp}.txt")

    summary_rows     = []
    all_detail_lines = []

    print(f"\n=== Variant B: LLM-based Paraphrase (Gemma 3 27B) ===")
    print(f"Subjects to process: {len(subject_ids)}")
    print(f"Output dir:          {args.output_dir}")
    print(f"Log dir:             {args.log_dir}")
    print(f"Timestamp:           {timestamp}\n")

    done_subjects    = 0
    skipped_subjects = 0

    for idx, participant_id in enumerate(subject_ids):
        print(f"\n--- [{idx+1}/{len(subject_ids)}] Participant {participant_id} ---")
        subject_start = time.time()

        transcript_path = os.path.join(args.data_dir, f"{participant_id}_TRANSCRIPT.csv")

        if not os.path.exists(transcript_path):
            print(f"  Transcript not found: {transcript_path} — skipping")
            skipped_subjects += 1
            summary_rows.append({
                "participant_id":  participant_id,
                "status":          "skipped_missing_file",
                "total_turns":     0,
                "turns_changed":   0,
                "turns_unchanged": 0,
                "elapsed_seconds": 0
            })
            continue

        try:
            df = pd.read_csv(transcript_path, sep='\t')
            df['speaker'] = df['speaker'].fillna('Unknown').astype(str)
            df['value']   = df['value'].fillna('').astype(str)
        except Exception as e:
            print(f"  Failed to load transcript: {e} — skipping")
            skipped_subjects += 1
            summary_rows.append({
                "participant_id":  participant_id,
                "status":          "skipped_load_error",
                "total_turns":     0,
                "turns_changed":   0,
                "turns_unchanged": 0,
                "elapsed_seconds": 0
            })
            continue

        participant_mask  = df['speaker'].str.strip().str.lower() == 'participant'
        participant_turns = df[participant_mask]
        total_turns       = len(participant_turns)
        print(f"  Participant turns to paraphrase: {total_turns}")

        turns_changed   = 0
        turns_unchanged = 0
        detail_lines    = [
            f"\n{'='*60}",
            f"Participant {participant_id}",
            f"{'='*60}"
        ]

        for turn_num, row_idx in enumerate(participant_turns.index, start=1):
            original_text    = df.at[row_idx, 'value']
            paraphrased_text = paraphrase_utterance(
                original_text, participant_id, base_url, model
            )

            df.at[row_idx, 'value'] = paraphrased_text

            changed = original_text.strip() != paraphrased_text.strip()
            if changed:
                turns_changed   += 1
            else:
                turns_unchanged += 1

            detail_lines.append(
                f"\n[Turn {turn_num}] {'CHANGED' if changed else 'SAME'}\n"
                f"  ORIG: {original_text}\n"
                f"  VARB: {paraphrased_text}"
            )

            if turn_num <= 3:
                print(f"  [Turn {turn_num}] ORIGINAL   : {original_text[:100]}")
                print(f"  [Turn {turn_num}] PARAPHRASED: {paraphrased_text[:100]}")

        output_path = os.path.join(args.output_dir, f"{participant_id}_TRANSCRIPT.csv")
        df.to_csv(output_path, sep='\t', index=False)

        elapsed = time.time() - subject_start
        print(f"  Saved: {output_path}")
        print(f"  Done in {elapsed:.1f}s | Changed: {turns_changed} | "
              f"Unchanged: {turns_unchanged}")

        summary_rows.append({
            "participant_id":  participant_id,
            "status":          "ok",
            "total_turns":     total_turns,
            "turns_changed":   turns_changed,
            "turns_unchanged": turns_unchanged,
            "elapsed_seconds": round(elapsed, 1)
        })
        all_detail_lines.extend(detail_lines)
        done_subjects += 1

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV written: {summary_path}")

    with open(detail_log_path, 'w') as f:
        f.write(f"Variant B Detailed Log — {timestamp}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Subjects processed: {done_subjects}\n")
        f.write(f"Subjects skipped:   {skipped_subjects}\n")
        f.write("\n".join(all_detail_lines))
    print(f"Detailed log written: {detail_log_path}")

    print("\n========================================")
    print("Variant B paraphrase generation complete")
    print(f"  Processed : {done_subjects}")
    print(f"  Skipped   : {skipped_subjects}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Log dir   : {args.log_dir}")
    print("========================================")

if __name__ == "__main__":
    main()
