#!/usr/bin/env python3

import os, glob, re, difflib
import numpy as np, pandas as pd

ROOT = "analysis_output"
BASELINE = os.path.join(ROOT, "Baseline")
VARIANT_C = os.path.join(ROOT, "VariantC")
PHQ8 = ["PHQ8_NoInterest","PHQ8_Depressed","PHQ8_Sleep","PHQ8_Tired",
        "PHQ8_Appetite","PHQ8_Failure","PHQ8_Concentrating","PHQ8_Moving"]
def sev_to_bin(s): return 1 if s >= 2 else 0

def _zs_only(paths):
    return [p for p in paths if not re.search(r'(fewshot|few_shot|few|_fs_|(^|_)fs[_.])',
            os.path.basename(p).lower())]
def find_meta(d):
    c=[]
    for pat in ["meta_review_zeroshot*.csv","meta_review_zs_*.csv","meta_review_zs*.csv"]:
        c+=glob.glob(os.path.join(d,"**",pat),recursive=True)
    if not c: c=glob.glob(os.path.join(d,"**","meta","*.csv"),recursive=True)
    c=_zs_only(c); return sorted(set(c))[0] if c else None
def find_quant(d):
    c=[]
    for pat in ["results_zero_shot*.csv","results_zs_*.csv","results_zs*.csv"]:
        c+=glob.glob(os.path.join(d,"**",pat),recursive=True)
    if not c:
        cc=glob.glob(os.path.join(d,"**","quan_zero_shot","*.csv"),recursive=True)
        c=[p for p in cc if os.path.basename(p).lower().startswith("results")] or cc
    c=_zs_only(c); return sorted(set(c))[0] if c else None

def load_sev(csv):
    df=pd.read_csv(csv); df["severity"]=pd.to_numeric(df["severity"],errors="coerce")
    return {int(r.participant_id):int(r.severity) for r in df.dropna(subset=["severity"]).itertuples()}
def load_items(csv):
    df=pd.read_csv(csv)
    for it in PHQ8: df[it]=pd.to_numeric(df[it],errors="coerce") if it in df.columns else np.nan
    return {int(r["participant_id"]):{it:r.get(it,np.nan) for it in PHQ8} for _,r in df.iterrows()}

bm, cm = find_meta(BASELINE), find_meta(VARIANT_C)
bq, cq = find_quant(BASELINE), find_quant(VARIANT_C)
print(f"baseline meta : {bm}\nvariantC meta : {cm}")
print(f"baseline quant: {bq}\nvariantC quant: {cq}\n")
B_sev, C_sev = load_sev(bm), load_sev(cm)
B_it,  C_it  = load_items(bq), load_items(cq)

flipped=[]
for pid in sorted(set(B_sev)&set(C_sev)):
    if sev_to_bin(B_sev[pid])!=sev_to_bin(C_sev[pid]):
        flipped.append(pid)
print(f"FLIPPED under C: {flipped}\n")

for pid in flipped:
    bb,cc = sev_to_bin(B_sev[pid]), sev_to_bin(C_sev[pid])
    lab={0:"not depressed",1:"depressed"}
    print(f"--- participant {pid} ---")
    print(f"  severity {B_sev[pid]} -> {C_sev[pid]}   diagnosis {lab[bb]} -> {lab[cc]}")
    if pid in B_it and pid in C_it:
        for it in PHQ8:
            b,c=B_it[pid][it],C_it[pid][it]
            if not (np.isnan(b) and np.isnan(c)) and not (b==c):
                print(f"    {it}: {b} -> {c}")
    hits = glob.glob(os.path.join(VARIANT_C,"**",f"*{pid}*"),recursive=True)
    txts = [h for h in hits if h.lower().endswith((".txt",".json",".csv")) and "result" not in os.path.basename(h).lower()]
    print(f"    files with this id under VariantC: {txts if txts else '(none found -- pull transcript manually)'}")
    print()
print("Next: for one flipped participant, open its baseline transcript and its VariantC")
print("transcript, and copy ONLY the turn(s) whose frequency phrase changed.")
