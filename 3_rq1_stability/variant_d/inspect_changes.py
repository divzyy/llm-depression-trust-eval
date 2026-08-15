

import os
import re
import random
import argparse
import pandas as pd
from difflib import SequenceMatcher

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

parser = argparse.ArgumentParser()
parser.add_argument('--rate',          type=int, required=True, choices=[10, 20, 50, 75])
parser.add_argument('--seed',          type=int, default=1, choices=[1, 2, 3, 4, 5])
parser.add_argument('--n',             type=int, default=12, help='Number of examples')
parser.add_argument('--participant',   type=int, default=None, help='Filter to one participant')
parser.add_argument('--compare-seeds', action='store_true',
                    help='Show same turns across seeds 1/2/3 side by side')
parser.add_argument('--first',         action='store_true',
                    help='Show first N changed turns instead of random sample')
parser.add_argument('--sample-seed',   type=int, default=42,
                    help='Seed for the random sampling itself (reproducible)')
args = parser.parse_args()

ORIGINAL_DIR = f"{DAIC}/transcripts"
TEST_BASE    = f"{ROOT}/rq1_perturbations/variant_d_test"

TEST_IDS = [316, 319, 330, 339, 345]
if args.participant:
    TEST_IDS = [args.participant]

def get_word_changes(orig_text, mod_text):
    orig_tokens = re.findall(r'\S+', orig_text)
    mod_tokens  = re.findall(r'\S+', mod_text)
    orig_lower  = [w.lower().strip('.,!?;:') for w in orig_tokens]
    mod_lower   = [w.lower().strip('.,!?;:') for w in mod_tokens]
    matcher = SequenceMatcher(None, orig_lower, mod_lower)
    changes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            changes.append((' '.join(orig_tokens[i1:i2]), ' '.join(mod_tokens[j1:j2])))
        elif tag == 'delete':
            changes.append((' '.join(orig_tokens[i1:i2]), '[DELETED]'))
        elif tag == 'insert':
            changes.append(('[INSERTED]', ' '.join(mod_tokens[j1:j2])))
    return changes

def highlight(orig_text, mod_text):
    orig_tokens = re.findall(r'\S+', orig_text)
    mod_tokens  = re.findall(r'\S+', mod_text)
    orig_lower  = [w.lower().strip('.,!?;:') for w in orig_tokens]
    mod_lower   = [w.lower().strip('.,!?;:') for w in mod_tokens]
    matcher = SequenceMatcher(None, orig_lower, mod_lower)
    o_out, m_out = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            o_out.extend(orig_tokens[i1:i2]); m_out.extend(mod_tokens[j1:j2])
        elif tag == 'replace':
            o_out.append(f">>{' '.join(orig_tokens[i1:i2])}<<")
            m_out.append(f">>{' '.join(mod_tokens[j1:j2])}<<")
        elif tag == 'delete':
            o_out.append(f">>{' '.join(orig_tokens[i1:i2])}<<")
        elif tag == 'insert':
            m_out.append(f">>{' '.join(mod_tokens[j1:j2])}<<")
    return ' '.join(o_out), ' '.join(m_out)

def load_transcript(path):
    df = pd.read_csv(path, sep='\t')
    df['speaker'] = df['speaker'].fillna('Unknown').astype(str)
    df['value']   = df['value'].fillna('').astype(str)
    return df

def show_single_seed():
    rate_dir = os.path.join(TEST_BASE, f"rate_{args.rate}", f"seed_{args.seed}")
    mode = "FIRST" if args.first else "RANDOM SAMPLE"
    print(f"\nInspecting changes — Rate={args.rate}%  Seed={args.seed}  [{mode}]")
    print(f"Participants: {TEST_IDS}")
    print(f"Changed words marked with >>word<<")
    print("=" * 70)

    all_changed = []  
    for pid in TEST_IDS:
        orig_path = os.path.join(ORIGINAL_DIR, f"{pid}_TRANSCRIPT.csv")
        mod_path  = os.path.join(rate_dir, f"{pid}_TRANSCRIPT.csv")
        if not os.path.exists(mod_path):
            continue
        orig = load_transcript(orig_path)
        mod  = load_transcript(mod_path)
        for i in range(min(len(orig), len(mod))):
            if orig.iloc[i]['speaker'].strip().lower() != 'participant':
                continue
            o = orig.iloc[i]['value']; m = mod.iloc[i]['value']
            if len(o.split()) < 3:
                continue
            if o != m:
                all_changed.append((pid, i, o, m))

    print(f"Total changed turns available: {len(all_changed)}")

    if args.first:
        chosen = all_changed[:args.n]
    else:
        rng = random.Random(args.sample_seed)
        chosen = rng.sample(all_changed, min(args.n, len(all_changed)))
        chosen.sort(key=lambda x: (x[0], x[1]))

    print()
    for pid, idx, o, m in chosen:
        changes = get_word_changes(o, m)
        o_hl, m_hl = highlight(o, m)
        print(f"Participant {pid}  |  Turn {idx}  |  {len(changes)} change(s)")
        print(f"  ORIGINAL : {o_hl}")
        print(f"  MODIFIED : {m_hl}")
        if changes:
            print(f"  CHANGES  : " + '  |  '.join(f"'{a}' → '{b}'" for a, b in changes))
        print()

    print("="*70)
    print(f"Showed {len(chosen)} of {len(all_changed)} changed turns (random sample).")

def show_compare_seeds():
    pid = TEST_IDS[0]
    print(f"\nCompare seeds — Rate={args.rate}%  Participant {pid}")
    print(f"Same turn shown under seed 1 / 2 / 3 — see how selection differs")
    print(f"Changed words marked with >>word<<")
    print("=" * 70)

    orig_path = os.path.join(ORIGINAL_DIR, f"{pid}_TRANSCRIPT.csv")
    orig = load_transcript(orig_path)

    seed_dfs = {}
    for s in [1, 2, 3]:
        p = os.path.join(TEST_BASE, f"rate_{args.rate}", f"seed_{s}", f"{pid}_TRANSCRIPT.csv")
        if os.path.exists(p):
            seed_dfs[s] = load_transcript(p)

    if not seed_dfs:
        print("No seed transcripts found.")
        return

    candidate_turns = []
    for i in range(len(orig)):
        if orig.iloc[i]['speaker'].strip().lower() != 'participant':
            continue
        o = orig.iloc[i]['value']
        if len(o.split()) < 3:
            continue
        changed_in_any = any(
            i < len(seed_dfs[s]) and seed_dfs[s].iloc[i]['value'] != o
            for s in seed_dfs
        )
        if changed_in_any:
            candidate_turns.append(i)

    rng = random.Random(args.sample_seed)
    chosen = rng.sample(candidate_turns, min(args.n, len(candidate_turns)))
    chosen.sort()

    print()
    for i in chosen:
        o = orig.iloc[i]['value']
        print(f"Turn {i}  ORIGINAL: {o}")
        for s in [1, 2, 3]:
            if s in seed_dfs and i < len(seed_dfs[s]):
                m = seed_dfs[s].iloc[i]['value']
                if m != o:
                    changes = get_word_changes(o, m)
                    swaps = '  '.join(f"'{a}'→'{b}'" for a, b in changes)
                    print(f"   seed {s}: {swaps}")
                else:
                    print(f"   seed {s}: (no change)")
        print()

    print("="*70)
    print(f"Showed {len(chosen)} turns across seeds 1/2/3.")
    print("Notice: each seed changes DIFFERENT words (or none) in the same turn.")

if args.compare_seeds:
    show_compare_seeds()
else:
    show_single_seed()
