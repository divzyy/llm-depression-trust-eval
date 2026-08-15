#!/usr/bin/env python3

import csv, re, os, glob
REF_META  = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/qual/meta_review_zeroshot_test_v2.csv")
REF_QUANT = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/quan/results_zero_shot_test41.csv")
ROOT = os.path.expanduser("~/ai-psychiatrist/analysis_output")
ITEMS=['PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired','PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating','PHQ8_Moving']
def num(x):
    m=re.search(r'-?\d+',str(x)); return int(m.group()) if m else None
def sev(p):
    o={}
    for r in csv.DictReader(open(p,newline='',encoding='utf-8',errors='replace')):
        a=num(r.get('participant_id')); s=num(r.get('severity'))
        if a is not None and s is not None: o[a]=s
    return o
def quant(p):
    o={}
    for r in csv.DictReader(open(p,newline='',encoding='utf-8',errors='replace')):
        a=num(r.get('participant_id'))
        if a is None: continue
        d={}
        for it in ITEMS:
            v=str(r.get(it,'')).strip()
            d[it]=None if v=='' or v.upper()=='N/A' or v.lower()=='none' else num(v)
        o[a]=d
    return o
def total(d): return sum(v for v in d.values() if v is not None)
rm=sev(REF_META); rq=quant(REF_QUANT)
boundary={p for p,s in rm.items() if s in (1,2)}
other={p for p,s in rm.items() if s not in (1,2)}
dq=glob.glob(f"{ROOT}/VariantD/rate_*/seed_*/quan_zero_shot/results_zero_shot*.csv")
def tally(group):
    up=down=same=0
    for qp in dq:
        pq=quant(qp)
        for p in group:
            if p in pq and p in rq:
                ch=total(pq[p])-total(rq[p])
                if ch>0: up+=1
                elif ch<0: down+=1
                else: same+=1
    return up,down,same
bu,bd,bs=tally(boundary)
ou,od,os_=tally(other)
print(f"boundary participants (ref sev 1 or 2), n={len(boundary)}:")
print(f"   score UP {bu}, DOWN {bd}, same {bs}   (up-down net {bu-bd:+d})")
print(f"non-boundary participants (ref sev 0,3,4), n={len(other)}:")
print(f"   score UP {ou}, DOWN {od}, same {os_}   (up-down net {ou-od:+d})")
