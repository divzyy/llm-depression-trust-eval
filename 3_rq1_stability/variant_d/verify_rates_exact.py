
import os
import re
import argparse
import pandas as pd
import statistics

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true',
                    help='Use full variant_d dir instead of variant_d_test')
parser.add_argument('--rate', type=int, choices=[10, 20, 50, 75], default=None)
parser.add_argument('--seed', type=int, choices=[1, 2, 3, 4, 5], default=None)
args = parser.parse_args()

ORIGINAL_DIR = f"{DAIC}/transcripts"

if args.full:
    VARIANT_DIR = f"{ROOT}/rq1_perturbations/variant_d"
    TEST_IDS = [
        316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383,
        385, 386, 389, 390, 393, 409, 413, 417, 422, 423, 427, 428, 430,
        436, 441, 445, 447, 449, 451, 455, 456, 459, 468, 472, 484, 485,
        487, 489
    ]
else:
    VARIANT_DIR = f"{ROOT}/rq1_perturbations/variant_d_test"
    TEST_IDS = [316, 319, 330, 339, 345]

RATES = [args.rate] if args.rate else [10, 20, 50, 75]
SEEDS = [args.seed] if args.seed else [1, 2, 3, 4, 5]

EXCLUDE_WORDS = {"participant", "ellie"}

def tokenize(text):
    """Identical tokenizer to the generator."""
    segments = re.split(r'(\b[a-zA-Z]+\b)', text)
    tokens = []
    for seg in segments:
        if re.match(r'^[a-zA-Z]+$', seg):
            tokens.append((seg, True))
        else:
            tokens.append((seg, False))
    return tokens

def get_eligible_indices(tokens):
    """Identical eligibility rules to the generator."""
    eligible = []
    for i, (tok, is_word) in enumerate(tokens):
        if not is_word:
            continue
        if tok.lower() in EXCLUDE_WORDS:
            continue
        # Skip vocal annotations <sneeze> <laughter>
        prev_tok = tokens[i-1][0] if i > 0 else ''
        next_tok = tokens[i+1][0] if i < len(tokens)-1 else ''
        if prev_tok.strip() == '<' and next_tok.strip() == '>':
            continue
        if "'" in prev_tok or "'" in next_tok:
            continue
        eligible.append(i)
    return eligible

def measure_turn(orig_text, mod_text):
    """
    Returns (changed_eligible, total_eligible) for one turn,
    comparing word positions directly.
    """
    orig_tokens = tokenize(orig_text)
    mod_tokens  = tokenize(mod_text)

    eligible_idx = get_eligible_indices(orig_tokens)

    changed = 0
    for idx in eligible_idx:
        orig_word = orig_tokens[idx][0]
        if idx < len(mod_tokens):
            mod_word = mod_tokens[idx][0]
        else:
            mod_word = ''
        if orig_word.lower() != mod_word.lower():
            changed += 1

    return changed, len(eligible_idx)

def main():
    print(f"EXACT rate verification (position-based)")
    print(f"Directory: {VARIANT_DIR}")
    print(f"Subjects: {len(TEST_IDS)}  |  Rates: {RATES}  |  Seeds: {SEEDS}")
    print("=" * 64)

    per_rate = {r: [] for r in RATES}

    for rate in RATES:
        for seed in SEEDS:
            seed_dir = os.path.join(VARIANT_DIR, f"rate_{rate}", f"seed_{seed}")
            if not os.path.isdir(seed_dir):
                continue

            for pid in TEST_IDS:
                orig_path = os.path.join(ORIGINAL_DIR, f"{pid}_TRANSCRIPT.csv")
                mod_path  = os.path.join(seed_dir, f"{pid}_TRANSCRIPT.csv")
                if not os.path.exists(mod_path):
                    continue

                orig = pd.read_csv(orig_path, sep='\t')
                mod  = pd.read_csv(mod_path,  sep='\t')
                orig['speaker'] = orig['speaker'].fillna('Unknown').astype(str)
                orig['value']   = orig['value'].fillna('').astype(str)
                mod['speaker']  = mod['speaker'].fillna('Unknown').astype(str)
                mod['value']    = mod['value'].fillna('').astype(str)

                turn_changed = 0
                turn_total   = 0
                for i in range(min(len(orig), len(mod))):
                    if orig.iloc[i]['speaker'].strip().lower() != 'participant':
                        continue
                    o = orig.iloc[i]['value']
                    m = mod.iloc[i]['value']
                    if len(o.split()) < 3:
                        continue
                    c, t = measure_turn(o, m)
                    turn_changed += c
                    turn_total   += t

                if turn_total > 0:
                    pct = round(turn_changed / turn_total * 100, 1)
                    per_rate[rate].append(pct)

    print(f"\n{'Rate':>6} {'Target':>8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Gap':>8} {'Status':>8}")
    print("-" * 64)
    for rate in RATES:
        vals = per_rate[rate]
        if not vals:
            print(f"{rate:>6}%  no data")
            continue
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        gap  = abs(mean - rate)
        status = "OK" if gap <= 5 else "OFF"
        print(f"{rate:>6}% {rate:>7}% {mean:>7.1f}% {std:>7.2f}% "
              f"{min(vals):>7.1f}% {max(vals):>7.1f}% {gap:>7.1f}% {status:>8}")

    print(f"\nNote: target = % of ELIGIBLE words (excludes speaker labels,")
    print(f"annotations, contractions) — the same set the generator selects from.")

if __name__ == "__main__":
    main()
