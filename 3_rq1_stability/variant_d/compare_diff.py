
import os
import re
import argparse
import pandas as pd
from difflib import SequenceMatcher

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

parser = argparse.ArgumentParser()
parser.add_argument('--participant', type=int, required=True)
parser.add_argument('--rate', type=int, choices=[10,20,50,75], default=None)
parser.add_argument('--seed', type=int, choices=[1,2,3,4,5], default=None)
args = parser.parse_args()

ORIGINAL_DIR = f"{DAIC}/transcripts"
TEST_BASE    = f"{ROOT}/rq1_perturbations/variant_d_test"

RATES = [args.rate] if args.rate else [10, 20, 50, 75]
SEEDS = [args.seed] if args.seed else [1, 2, 3]

PID = args.participant

def load(path):
    df = pd.read_csv(path, sep='\t')
    df['speaker'] = df['speaker'].fillna('Unknown').astype(str)
    df['value']   = df['value'].fillna('').astype(str)
    return df

def word_swaps(o, m):
    ot = o.split(); mt = m.split()
    sm = SequenceMatcher(None, [w.lower() for w in ot], [w.lower() for w in mt])
    swaps = []
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag == 'replace':
            swaps.append(f"'{' '.join(ot[i1:i2])}' -> '{' '.join(mt[j1:j2])}'")
        elif tag == 'delete':
            swaps.append(f"'{' '.join(ot[i1:i2])}' -> [removed]")
        elif tag == 'insert':
            swaps.append(f"[added] -> '{' '.join(mt[j1:j2])}'")
    return swaps

orig_path = os.path.join(ORIGINAL_DIR, f"{PID}_TRANSCRIPT.csv")
orig = load(orig_path)

lines_out = []
def emit(s=""):
    print(s)
    lines_out.append(s)

emit(f"Comparison for participant {PID}")
emit(f"Original: {orig_path}")
emit("="*72)

for rate in RATES:
    for seed in SEEDS:
        mod_path = os.path.join(TEST_BASE, f"rate_{rate}", f"seed_{seed}", f"{PID}_TRANSCRIPT.csv")
        if not os.path.exists(mod_path):
            continue
        mod = load(mod_path)

        emit("")
        emit("#"*72)
        emit(f"# RATE {rate}%   SEED {seed}")
        emit("#"*72)

        n_changed = 0
        total_word_changes = 0
        for i in range(min(len(orig), len(mod))):
            if orig.iloc[i]['speaker'].strip().lower() != 'participant':
                continue
            o = orig.iloc[i]['value']
            m = mod.iloc[i]['value']
            if o == m:
                continue
            n_changed += 1
            swaps = word_swaps(o, m)
            total_word_changes += len(swaps)
            st = orig.iloc[i]['start_time']
            emit(f"\n[turn {i}, time {st}]")
            emit(f"  ORIG: {o}")
            emit(f"  MOD : {m}")
            emit(f"  SWAP: " + "  |  ".join(swaps))

        emit(f"\n  >> {n_changed} turns changed, {total_word_changes} word-level swaps")

out_file = os.path.join(TEST_BASE, f"comparison_{PID}.txt")
with open(out_file, 'w') as f:
    f.write("\n".join(lines_out))
emit("")
emit("="*72)
emit(f"Saved full report to: {out_file}")
