#!/usr/bin/env python3


import csv, re, os

REF_META  = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/qual/meta_review_zeroshot_test_v2.csv")
REF_QUANT = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/quan/results_zero_shot_test41.csv")
C_META    = os.path.expanduser("~/ai-psychiatrist/analysis_output/VariantC/meta/meta_review_zeroshot_varC.csv")
C_QUANT   = os.path.expanduser("~/ai-psychiatrist/analysis_output/VariantC/quan/results_zero_shot_varC.csv")
TRAIN = os.path.expanduser("~/daic_woz_data/labels/train_split_Depression_AVEC2017.csv")
DEV   = os.path.expanduser("~/daic_woz_data/labels/dev_split_Depression_AVEC2017.csv")
CUT = 2
ITEMS=['PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired','PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating','PHQ8_Moving']

def num(x):
    m=re.search(r'-?\d+', str(x)); return int(m.group()) if m else None
def sev(path):
    o={}
    for r in csv.DictReader(open(path,newline='',encoding='utf-8',errors='replace')):
        p=num(r.get('participant_id')); s=num(r.get('severity'))
        if p is not None and s is not None: o[p]=s
    return o
def quant(path):
    o={}
    for r in csv.DictReader(open(path,newline='',encoding='utf-8',errors='replace')):
        p=num(r.get('participant_id'))
        if p is None: continue
        d={}
        for it in ITEMS:
            v=str(r.get(it,'')).strip()
            d[it]=None if v=='' or v.upper()=='N/A' or v.lower()=='none' else num(v)
        o[p]=d
    return o
def labels(*ps):
    g={}
    for pth in ps:
        if not os.path.exists(pth): continue
        for r in csv.DictReader(open(pth,newline='',encoding='utf-8',errors='replace')):
            pid=lab=None
            for k,v in r.items():
                kl=k.lower().strip()
                if kl in ('participant_id','participant'): pid=num(v)
                if 'binary' in kl: lab=num(v)
            if pid is not None and lab is not None: g[pid]=lab
    return g
def b(s): return 1 if s>=CUT else 0

rm,rq = sev(REF_META), quant(REF_QUANT)
cm,cq = sev(C_META), quant(C_QUANT)
gt = labels(TRAIN,DEV)

print("=== DIAGNOSIS-LEVEL direction under C ===")
harm=help_=0
for p in rm:
    if p in cm and b(rm[p])!=b(cm[p]):
        if p in gt:
            if b(rm[p])==gt[p]: harm+=1; tag="harmful correct->wrong"
            else: help_+=1; tag="HELPFUL wrong->correct"
            print(f"  {p}: ref_sev {rm[p]} -> C_sev {cm[p]}, gt {gt[p]}  [{tag}]")
print(f"  totals: harmful {harm}, helpful {help_}\n")

print("=== SEVERITY moved toward/away from the correct side (no flip needed) ===")
for p in sorted(rm):
    if p in cm and p in gt:
        r,c=rm[p],cm[p]
        if r==c: continue
        print(f"  {p}: ref_sev {r} (bin {b(r)}) -> C_sev {c} (bin {b(c)}), gt {gt[p]}")
print()

print("=== TOTAL PHQ-8 score: did C move it toward the true label direction? ===")
def total(d): return sum(v for v in d.values() if v is not None)
for p in sorted(rq):
    if p in cq:
        rt,ct=total(rq[p]),total(cq[p])
        if rt!=ct:
            print(f"  {p}: total {rt} -> {ct} (change {ct-rt:+d}), gt {gt.get(p,'?')}")
