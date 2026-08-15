#!/usr/bin/env python3

import csv, re, os
REF_QUANT = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/quan/results_zero_shot_test41.csv")
C_QUANT   = os.path.expanduser("~/ai-psychiatrist/analysis_output/VariantC/quan/results_zero_shot_varC.csv")
TRAIN = os.path.expanduser("~/daic_woz_data/labels/train_split_Depression_AVEC2017.csv")
DEV   = os.path.expanduser("~/daic_woz_data/labels/dev_split_Depression_AVEC2017.csv")
ITEMS=['PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired','PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating','PHQ8_Moving']

def num(x):
    m=re.search(r'-?\d+', str(x)); return int(m.group()) if m else None
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
def total(d): return sum(v for v in d.values() if v is not None)

rq,cq,gt = quant(REF_QUANT), quant(C_QUANT), labels(TRAIN,DEV)
up_gt1=up_gt0=down_gt1=down_gt0=same=0
for p in rq:
    if p not in cq or p not in gt: continue
    ch = total(cq[p]) - total(rq[p])
    if ch==0: same+=1
    elif ch>0:
        if gt[p]==1: up_gt1+=1     # up, truly depressed -> toward correct
        else:        up_gt0+=1     # up, truly not depressed -> away (over-diagnosis)
    else:
        if gt[p]==1: down_gt1+=1   # down, truly depressed -> away
        else:        down_gt0+=1   # down, truly not depressed -> toward correct
print(f"score went UP:   {up_gt1+up_gt0}   (toward truth if depressed: {up_gt1}, away if not-depressed: {up_gt0})")
print(f"score went DOWN: {down_gt1+down_gt0}   (away if depressed: {down_gt1}, toward truth if not-depressed: {down_gt0})")
print(f"unchanged: {same}")
print()
print(f"moved TOWARD true label: {up_gt1+down_gt0}")
print(f"moved AWAY from true label: {up_gt0+down_gt1}")
