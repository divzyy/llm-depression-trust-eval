#!/usr/bin/env python3

import csv, re, os, glob
REF_META = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/qual/meta_review_zeroshot_test_v2.csv")
ROOT = os.path.expanduser("~/ai-psychiatrist/analysis_output")
CUT=2
def num(x):
    m=re.search(r'-?\d+',str(x)); return int(m.group()) if m else None
def sev(p):
    o={}
    for r in csv.DictReader(open(p,newline='',encoding='utf-8',errors='replace')):
        a=num(r.get('participant_id')); s=num(r.get('severity'))
        if a is not None and s is not None: o[a]=s
    return o
def b(s): return 1 if s>=CUT else 0
rm=sev(REF_META)

def direction(name, meta_files):
    up=down=0; up_ids=[]; down_ids=[]
    for mp in meta_files:
        pm=sev(mp)
        for p in rm:
            if p in pm and b(rm[p])!=b(pm[p]):
                if b(rm[p])==0: up+=1; up_ids.append(p)      # notdep -> dep
                else:           down+=1; down_ids.append(p)  # dep -> notdep
    print(f"### {name}")
    print(f"  up (not-dep -> dep):  {up}   ids: {sorted(set(up_ids))}")
    print(f"  down (dep -> not-dep):{down}   ids: {sorted(set(down_ids))}")
    print()

direction("A: all symptoms", glob.glob(f"{ROOT}/VariantA/All/meta/meta_review_zs_*.csv"))
direction("B: whole transcript", glob.glob(f"{ROOT}/VariantB/meta/meta_review_zeroshot_test_v2.csv"))
direction("C: frequency phrases", glob.glob(f"{ROOT}/VariantC/meta/meta_review_zeroshot_varC.csv"))
direction("D: random (pooled)", glob.glob(f"{ROOT}/VariantD/rate_*/seed_*/meta/meta_review_zeroshot*.csv"))
