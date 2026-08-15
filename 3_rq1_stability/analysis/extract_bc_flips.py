#!/usr/bin/env python3


import os, glob, csv, re, sys


VARIANT_ROOT = os.path.expanduser("~/ai-psychiatrist/analysis_output")
REF_CSV = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/qual/meta_review_zeroshot_test_v2.csv")
TRAIN_LABELS = os.path.expanduser("~/daic_woz_data/labels/train_split_Depression_AVEC2017.csv")
DEV_LABELS   = os.path.expanduser("~/daic_woz_data/labels/dev_split_Depression_AVEC2017.csv")
DEPRESSED_CUT = 2     

D_HARMFUL_SUBJECTS = {385,484,339,362,423,417,459,409,451,430,422}

def to_int(x):
    if x is None: return None
    m = re.search(r"-?\d+", str(x))
    return int(m.group()) if m else None

def read_meta(path):
    out = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            pid = to_int(row.get("participant_id"))
            sev = to_int(row.get("severity"))
            if pid is not None and sev is not None:
                out[pid] = sev
    return out

def read_labels(*paths):
    gt = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"WARNING: label file missing: {p}", file=sys.stderr); continue
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                pid = lab = None
                for k, v in row.items():
                    kl = k.lower().strip()
                    if kl in ("participant_id","participant"): pid = to_int(v)
                    if "binary" in kl: lab = to_int(v)
                if pid is not None and lab is not None:
                    gt[pid] = lab
    return gt

def binary(s): return 1 if s >= DEPRESSED_CUT else 0

def find_meta(variant_letter):
    
    vdir = os.path.join(VARIANT_ROOT, f"Variant{variant_letter}")
    if not os.path.isdir(vdir):
        print(f"WARNING: {vdir} not found", file=sys.stderr); return None
    cands = glob.glob(os.path.join(vdir, "**", "meta_review_zeroshot**.csv"), recursive=True)
    if not cands:  # fall back to any meta file
        cands = glob.glob(os.path.join(vdir, "**", "*meta*zeroshot*.csv"), recursive=True) \
              or glob.glob(os.path.join(vdir, "**", "*meta*.csv"), recursive=True)
    if not cands:
        print(f"WARNING: no meta file found under {vdir}", file=sys.stderr); return None
    if len(cands) > 1:
        print(f"NOTE: multiple meta candidates for Variant{variant_letter}, using first:")
        for c in cands: print("     ", c)
    return cands[0]

ref = read_meta(REF_CSV)
gt  = read_labels(TRAIN_LABELS, DEV_LABELS)
print(f"Reference run: {len(ref)} participants | GT labels: {len(gt)}\n")

for letter in ("B", "C"):
    path = find_meta(letter)
    if not path:
        continue
    pert = read_meta(path)
    common = [p for p in ref if p in pert]
    missing = sorted(p for p in ref if p not in pert)
    flips = []
    for pid in common:
        rb, pb = binary(ref[pid]), binary(pert[pid])
        if rb != pb:
            if pid in gt:
                direction = "harmful(correct->wrong)" if rb == gt[pid] else "helpful(wrong->correct)"
            else:
                direction = "unknown_no_GT"
            flips.append((pid, ref[pid], pert[pid], rb, pb, gt.get(pid,""), direction))

    print(f"PERTURBATION {letter}")
    print(f"file: {path}")
    print(f"participants compared: {len(common)}  (missing from run: {len(missing)} {missing})")
    harmful = [f for f in flips if f[6].startswith('harmful')]
    helpful = [f for f in flips if f[6].startswith('helpful')]
    print(f"TOTAL FLIPS: {len(flips)}   harmful: {len(harmful)}   helpful: {len(helpful)}")
    if flips:
        print("  pid  ref_sev pert_sev ref_bin pert_bin gt  direction   in_D_fragile?")
        for pid,rs,ps,rb,pb,g,d in sorted(flips):
            tag = "YES" if pid in D_HARMFUL_SUBJECTS else "no"
            print(f"  {pid:<4} {rs:^7} {ps:^8} {rb:^7} {pb:^8} {str(g):^3} {d:<24} {tag}")
        sevs = sorted(set(rs for _,rs,_,_,_,_,_ in flips))
        print(f"  reference severities of flippers: {sevs}  (boundary = 1 and 2)")
        overlap = sorted(set(f[0] for f in flips) & D_HARMFUL_SUBJECTS)
        print(f"  overlap with D's fragile subjects: {overlap if overlap else 'none'}")
    print()
