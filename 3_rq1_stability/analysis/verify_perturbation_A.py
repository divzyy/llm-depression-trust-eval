#!/usr/bin/env python3

import csv, re, os, glob

REF_META = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/qual/meta_review_zeroshot_test_v2.csv")
VARIANT_A = os.path.expanduser("~/ai-psychiatrist/analysis_output/VariantA")
TRAIN_LABELS = os.path.expanduser("~/daic_woz_data/labels/train_split_Depression_AVEC2017.csv")
DEV_LABELS   = os.path.expanduser("~/daic_woz_data/labels/dev_split_Depression_AVEC2017.csv")
DEPRESSED_CUT = 2

PHQ_NAMES = {
    'PHQ_1':'Loss of interest', 'PHQ_2':'Depressed mood', 'PHQ_3':'Sleep',
    'PHQ_4':'Fatigue', 'PHQ_5':'Appetite', 'PHQ_6':'Feelings of failure',
    'PHQ_7':'Concentration', 'PHQ_8':'Movement', 'All':'All symptoms together',
}

def num(x):
    m = re.search(r'-?\d+', str(x)); return int(m.group()) if m else None

def read_sev(path):
    out={}
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            pid=num(row.get('participant_id')); sev=num(row.get('severity'))
            if pid is not None and sev is not None: out[pid]=sev
    return out

def read_labels(*paths):
    gt={}
    for p in paths:
        if not os.path.exists(p): continue
        with open(p, newline='', encoding='utf-8', errors='replace') as f:
            for row in csv.DictReader(f):
                pid=lab=None
                for k,v in row.items():
                    kl=k.lower().strip()
                    if kl in ('participant_id','participant'): pid=num(v)
                    if 'binary' in kl: lab=num(v)
                if pid is not None and lab is not None: gt[pid]=lab
    return gt

def binary(s): return 1 if s>=DEPRESSED_CUT else 0

ref = read_sev(REF_META)
gt  = read_labels(TRAIN_LABELS, DEV_LABELS)
print(f"Reference: {len(ref)} participants | GT: {len(gt)}\n")
print(f"{'Symptom':<24}{'n':>4}{'flips':>7}{'harmful':>9}{'helpful':>9}   flipped_ids")
print("-"*70)

for folder in ['All','PHQ_1','PHQ_2','PHQ_3','PHQ_4','PHQ_5','PHQ_6','PHQ_7','PHQ_8']:
    matches = glob.glob(os.path.join(VARIANT_A, folder, "meta", "meta_review_zs_*.csv"))
    name = PHQ_NAMES.get(folder, folder)
    if not matches:
        print(f"{name:<24}{'--':>4}   (no zero-shot meta file found)")
        continue
    pert = read_sev(matches[0])
    common = [p for p in ref if p in pert]
    flips=[]; harmful=helpful=0
    for p in common:
        rb, pb = binary(ref[p]), binary(pert[p])
        if rb!=pb:
            flips.append(p)
            if p in gt:
                if rb==gt[p]: harmful+=1
                else: helpful+=1
    rate = len(flips)/len(common) if common else 0
    print(f"{name:<24}{len(common):>4}{len(flips):>7}{harmful:>9}{helpful:>9}   {sorted(flips)}")
