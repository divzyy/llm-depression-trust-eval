

import os
import re
import pandas as pd
import argparse
from difflib import SequenceMatcher

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

parser = argparse.ArgumentParser()
parser.add_argument('--rate', type=int, choices=[10, 20, 50, 75], default=None,
                    help='Only process this rate (default: all)')
parser.add_argument('--seed', type=int, choices=[1, 2, 3, 4, 5], default=None,
                    help='Only process this seed (default: all)')
args = parser.parse_args()

ORIGINAL_DIR  = f"{DAIC}/transcripts"
VARIANT_D_DIR = f"{ROOT}/rq1_perturbations/variant_d"
LOG_DIR       = os.path.join(VARIANT_D_DIR, "diff_logs")
os.makedirs(LOG_DIR, exist_ok=True)

RATES = [args.rate] if args.rate else [10, 20, 50, 75]
SEEDS = [args.seed] if args.seed else [1, 2, 3, 4, 5]

TEST_IDS = [
    316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383,
    385, 386, 389, 390, 393, 409, 413, 417, 422, 423, 427, 428, 430,
    436, 441, 445, 447, 449, 451, 455, 456, 459, 468, 472, 484, 485,
    487, 489
]

def tokenize(text):
    """
    Split text into word tokens (lowercase, alphabetic only for comparison).
    Keeps original tokens for display.
    """
    return re.findall(r'\S+', text)

def word_diff(original_text, modified_text):
    """
    Compare two strings at the word level using SequenceMatcher.

    Returns:
        words_removed : words present in original but replaced/removed
        words_added   : words present in modified but not in original
        n_changed     : number of word positions that differ
        pct_changed   : percentage of original words that changed
    """
    orig_tokens = tokenize(original_text)
    mod_tokens  = tokenize(modified_text)

    orig_lower = [w.lower().strip('.,!?;:') for w in orig_tokens]
    mod_lower  = [w.lower().strip('.,!?;:') for w in mod_tokens]

    matcher = SequenceMatcher(None, orig_lower, mod_lower)

    words_removed = []
    words_added   = []
    n_changed     = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            words_removed.extend(orig_tokens[i1:i2])
            words_added.extend(mod_tokens[j1:j2])
            n_changed += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            words_removed.extend(orig_tokens[i1:i2])
            n_changed += (i2 - i1)
        elif tag == 'insert':
            words_added.extend(mod_tokens[j1:j2])
            n_changed += (j2 - j1)

    n_orig      = len(orig_tokens)
    pct_changed = round(n_changed / n_orig * 100, 1) if n_orig > 0 else 0.0

    return (
        ' | '.join(words_removed),
        ' | '.join(words_added),
        n_changed,
        n_orig,
        pct_changed
    )



def process_rate_seed(rate, seed):
    """
    Generate diff log for one rate/seed combination.
    Returns list of turn-level dicts and list of summary dicts.
    """
    turn_rows    = []
    summary_rows = []

    print(f"\n--- rate_{rate} / seed_{seed} ---")

    for participant_id in TEST_IDS:
        orig_path = os.path.join(ORIGINAL_DIR, f"{participant_id}_TRANSCRIPT.csv")
        mod_path  = os.path.join(VARIANT_D_DIR, f"rate_{rate}", f"seed_{seed}",
                                 f"{participant_id}_TRANSCRIPT.csv")

        if not os.path.exists(mod_path):
            print(f"  [{participant_id}] MISSING — skipping")
            continue

        try:
            orig = pd.read_csv(orig_path, sep='\t')
            mod  = pd.read_csv(mod_path,  sep='\t')
            orig['speaker'] = orig['speaker'].fillna('Unknown').astype(str)
            orig['value']   = orig['value'].fillna('').astype(str)
            mod['speaker']  = mod['speaker'].fillna('Unknown').astype(str)
            mod['value']    = mod['value'].fillna('').astype(str)
        except Exception as e:
            print(f"  [{participant_id}] LOAD ERROR: {e}")
            continue

        turns_changed   = 0
        turns_total_eligible = 0
        total_words_changed  = 0
        total_words_orig     = 0

        for idx in range(min(len(orig), len(mod))):
            orig_speaker = orig.iloc[idx]['speaker'].strip().lower()
            if orig_speaker != 'participant':
                continue

            orig_text = orig.iloc[idx]['value']
            mod_text  = mod.iloc[idx]['value']

            if len(orig_text.split()) < 3:
                continue

            turns_total_eligible += 1
            changed = (orig_text != mod_text)

            if changed:
                turns_changed += 1
                words_removed, words_added, n_changed, n_orig, pct = word_diff(
                    orig_text, mod_text
                )
                total_words_changed += n_changed
                total_words_orig    += n_orig
            else:
                words_removed = ''
                words_added   = ''
                n_changed     = 0
                n_orig        = len(tokenize(orig_text))
                pct           = 0.0
                total_words_orig += n_orig

            turn_rows.append({
                'participant_id': participant_id,
                'rate':           rate,
                'seed':           seed,
                'turn_idx':       idx,
                'changed':        changed,
                'original_text':  orig_text,
                'modified_text':  mod_text,
                'words_removed':  words_removed,
                'words_added':    words_added,
                'n_words_orig':   n_orig,
                'n_words_changed': n_changed,
                'pct_changed':    pct
            })

        overall_pct = round(total_words_changed / total_words_orig * 100, 1) \
                      if total_words_orig > 0 else 0.0

        summary_rows.append({
            'participant_id':         participant_id,
            'rate':                   rate,
            'seed':                   seed,
            'eligible_turns':         turns_total_eligible,
            'turns_changed':          turns_changed,
            'turns_unchanged':        turns_total_eligible - turns_changed,
            'pct_turns_changed':      round(turns_changed / turns_total_eligible * 100, 1)
                                      if turns_total_eligible > 0 else 0.0,
            'total_words_orig':       total_words_orig,
            'total_words_changed':    total_words_changed,
            'overall_pct_word_change': overall_pct
        })

        print(f"  [{participant_id}] {turns_changed}/{turns_total_eligible} turns changed "
              f"({overall_pct}% words changed)")

    return turn_rows, summary_rows

def main():
    print(f"Generating diff logs for Variant D")
    print(f"Rates: {RATES}  |  Seeds: {SEEDS}")
    print(f"Output: {LOG_DIR}")
    print("=" * 70)

    all_summary_rows = []

    for rate in RATES:
        for seed in SEEDS:
            turn_rows, summary_rows = process_rate_seed(rate, seed)
            all_summary_rows.extend(summary_rows)

            if not turn_rows:
                print(f"  No data for rate_{rate}/seed_{seed} — skipping save")
                continue

        
            turn_df = pd.DataFrame(turn_rows)
            log_path = os.path.join(LOG_DIR, f"diff_log_rate_{rate}_seed_{seed}.csv")
            turn_df.to_csv(log_path, index=False)
            print(f"  Saved: {log_path} ({len(turn_df)} turns)")

    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)
        summary_path = os.path.join(LOG_DIR, "diff_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\nSummary saved: {summary_path}")
        print("\n--- Mean word change % vs target ---")
        agg = summary_df.groupby('rate').agg(
            mean_word_pct=('overall_pct_word_change', 'mean'),
            std_word_pct=('overall_pct_word_change', 'std'),
            mean_turns_pct=('pct_turns_changed', 'mean'),
        ).round(2)
        agg['target_pct'] = agg.index
        print(agg.to_string())

    print(f"\nDone.")

if __name__ == "__main__":
    main()
