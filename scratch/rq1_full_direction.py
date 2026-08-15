#!/usr/bin/env python3

import csv, re, os, glob
REF_META  = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/qual/meta_review_zeroshot_test_v2.csv")
REF_QUANT = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/quan/results_zero_shot_test41.csv")
TRAIN = os.path.expanduser("~/daic_woz_data/labels/train_split_Depression_AVEC2017.csv")
DEV   = os.path.expanduser("~/daic_woz_data/labels/dev_split_Depression_AVEC2017.csv")
ROOT  = os.path.expanduser("~/ai-psychiatrist/analysis_output")
CUT=2
ITEMS=['PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired','PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating','PHQ8_Moving']

def num(x):
    m=re.search(r'-?\d+',str(x)); return int(m.group()) if m else None
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
def total(d): return sum(v for v in d.values() if v is not None)

rm,rq,gt = sev(REF_META), quant(REF_QUANT), labels(TRAIN,DEV)

def analyse(name, meta_files, quant_files):
    cm={}; cq={}
    flips_h=flips_help=0
    tot_toward=tot_away=tot_same=0
    up=down=0
    n_meta=n_quant=0
    for mp in meta_files:
        pm=sev(mp)
        for p in rm:
            if p in pm:
                n_meta+=1
                if b(rm[p])!=b(pm[p]):
                    if p in gt:
                        if b(rm[p])==gt[p]: flips_h+=1
                        else: flips_help+=1
    for qp in quant_files:
        pq=quant(qp)
        for p in rq:
            if p in pq and p in gt:
                n_quant+=1
                ch=total(pq[p])-total(rq[p])
                if ch>0: up+=1
                elif ch<0: down+=1
                if ch==0: tot_same+=1
                elif (ch>0 and gt[p]==1) or (ch<0 and gt[p]==0): tot_toward+=1
                else: tot_away+=1
    print(f"### {name}")
    print(f"  diagnosis flips: harmful(correct->wrong) {flips_h}, helpful(wrong->correct) {flips_help}")
    print(f"  score changes: up {up}, down {down}, unchanged {tot_same}")
    print(f"  score direction vs GT: toward {tot_toward}, away {tot_away}")
    print()

analyse("A: all symptoms together",
        glob.glob(f"{ROOT}/VariantA/All/meta/meta_review_zs_*.csv"),
        glob.glob(f"{ROOT}/VariantA/All/quan_zero_shot/results_zero_shot*.csv"))
analyse("B: whole transcript",
        glob.glob(f"{ROOT}/VariantB/meta/meta_review_zeroshot*.csv"),
        glob.glob(f"{ROOT}/VariantB/quan_zero_shot/results_zero_shot*.csv"))
analyse("C: frequency phrases",
        glob.glob(f"{ROOT}/VariantC/meta/meta_review_zeroshot_varC.csv"),
        glob.glob(f"{ROOT}/VariantC/quan/results_zero_shot_varC.csv"))
# D (pooled over all runs)
analyse("D: random replacement (pooled)",
        glob.glob(f"{ROOT}/VariantD/rate_*/seed_*/meta/meta_review_zeroshot*.csv"),
        glob.glob(f"{ROOT}/VariantD/rate_*/seed_*/quan_zero_shot/results_zero_shot*.csv"))
