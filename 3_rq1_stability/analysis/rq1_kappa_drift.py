#!/usr/bin/env python3

import os, glob, csv, re, sys
from statistics import mean

# CONFIG
ANALYSIS_ROOT = os.path.expanduser("~/ai-psychiatrist/analysis_output")
REF_QUANT = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/quan/results_zero_shot_test41.csv")
REF_META  = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/qual/meta_review_zeroshot_test_v2.csv")
DEPRESSED_CUT = 2
ITEMS = ['PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired',
         'PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating','PHQ8_Moving']
NA_IN_TOTAL = 0.0   

def num_or_none(x):
    if x is None: return None
    s = str(x).strip()
    if s == '' or s.upper() == 'N/A' or s.lower() == 'none': return None
    m = re.search(r'-?\d+(\.\d+)?', s)
    return float(m.group()) if m else None

def read_quant(path):
    
    out = {}
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            pid = num_or_none(row.get('participant_id'))
            if pid is None: continue
            out[int(pid)] = {it: num_or_none(row.get(it)) for it in ITEMS}
    return out

def read_severity(path):
    out = {}
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            pid = num_or_none(row.get('participant_id'))
            sev = num_or_none(row.get('severity'))
            if pid is not None and sev is not None:
                out[int(pid)] = int(sev)
    return out

def binary(sev): return 1 if sev >= DEPRESSED_CUT else 0

def cohen_kappa(pairs):

    n = len(pairs)
    if n == 0: return None
    po = sum(1 for a,b in pairs if a==b) / n
    ref1 = sum(a for a,_ in pairs)/n; ref0 = 1-ref1
    prt1 = sum(b for _,b in pairs)/n; prt0 = 1-prt1
    pe = ref1*prt1 + ref0*prt0
    if pe == 1: return 1.0
    return (po - pe) / (1 - pe)

def total(items):
    
    return sum(v if v is not None else NA_IN_TOTAL for v in items.values())

def find(variant, kind):
    
    vdir = os.path.join(ANALYSIS_ROOT, f"Variant{variant}")
    if kind == 'quant':
        pats = ["**/results_zero_shot*.csv"]
    else:
        pats = ["**/meta_review_zeroshot*.csv", "**/meta_review*varC.csv", "**/*meta*.csv"]
    for p in pats:
        c = sorted(glob.glob(os.path.join(vdir, p), recursive=True))
        c = [x for x in c if 'fewshot' not in x.lower() and 'few_shot' not in x.lower()]
        if c:
            if len(c) > 1:
                print(f"  NOTE multiple {kind} files for Variant{variant}, using first:")
                for x in c: print("     ", x)
            return c[0]
    return None

def rate_seed_key(path):
    m = re.search(r'(rate_[^/]+/seed_[^/]+)', path)
    return m.group(1) if m else path

ref_q = read_quant(REF_QUANT)
ref_m = read_severity(REF_META)
print(f"Reference: {len(ref_q)} participants (quant), {len(ref_m)} (meta)")
print(f"  quant: {REF_QUANT}")
print(f"  meta : {REF_META}\n")

def report(name, quant_path, meta_path):
    if not quant_path or not os.path.exists(quant_path):
        print(f"{name}: quant file not found, skipping\n"); return
    if not meta_path or not os.path.exists(meta_path):
        print(f"{name}: meta file not found, skipping\n"); return
    pq = read_quant(quant_path); pm = read_severity(meta_path)

    kp = [(binary(ref_m[p]), binary(pm[p])) for p in ref_m if p in pm]
    kappa = cohen_kappa(kp)

    per_item_abs = []; total_abs = []; na_ref=[]; na_pert=[]
    common = [p for p in ref_q if p in pq]
    for p in common:
        r, q = ref_q[p], pq[p]
        for it in ITEMS:
            if r[it] is not None and q[it] is not None:
                per_item_abs.append(abs(r[it]-q[it]))
        total_abs.append(abs(total(r)-total(q)))
        na_ref.append(sum(1 for it in ITEMS if r[it] is None))
        na_pert.append(sum(1 for it in ITEMS if q[it] is None))

    print(f"------------- {name} ------------")
    print(f"quant file: {quant_path}")
    print(f"meta  file: {meta_path}")
    print(f"participants (meta pairs): {len(kp)}   (quant common: {len(common)})")
    print(f"Cohen's kappa (binary diagnosis): {kappa:.3f}" if kappa is not None else "kappa: n/a")
    if per_item_abs:
        print(f"mean per-item drift (jointly-scored items only): {mean(per_item_abs):.3f}  (n items = {len(per_item_abs)})")
    if total_abs:
        print(f"mean summed-total drift (N/A={NA_IN_TOTAL} in total):   {mean(total_abs):.3f}")
    print(f"mean N/A items per participant: ref {mean(na_ref):.2f} -> pert {mean(na_pert):.2f}")
    changed_na = sum(1 for a,b in zip(na_ref,na_pert) if a!=b)
    print(f"participants whose N/A count changed: {changed_na}/{len(common)}\n")

report("PERTURBATION B", find('B','quant'), find('B','meta'))
report("PERTURBATION C", find('C','quant'), find('C','meta'))


d_quant = sorted(glob.glob(os.path.join(ANALYSIS_ROOT, "VariantD", "rate_*", "seed_*", "**", "results_zero_shot*.csv"), recursive=True))
d_meta  = sorted(glob.glob(os.path.join(ANALYSIS_ROOT, "VariantD", "rate_*", "seed_*", "**", "meta_review_zeroshot*.csv"), recursive=True))

if not d_quant:
    print("PERTURBATION D: no quant files found under VariantD/rate_*/seed_*/ , skipping")
else:
    meta_by_key = {rate_seed_key(m): m for m in d_meta}
    kp=[]; per_item_abs=[]; total_abs=[]; na_ref=[]; na_pert=[]; runs=0; unpaired=0
    for qpath in d_quant:
        mpath = meta_by_key.get(rate_seed_key(qpath))
        if not mpath:
            unpaired += 1; continue
        runs += 1
        pq = read_quant(qpath); pm = read_severity(mpath)
        for p in ref_m:
            if p in pm: kp.append((binary(ref_m[p]), binary(pm[p])))
        for p in ref_q:
            if p in pq:
                r,q = ref_q[p], pq[p]
                for it in ITEMS:
                    if r[it] is not None and q[it] is not None:
                        per_item_abs.append(abs(r[it]-q[it]))
                total_abs.append(abs(total(r)-total(q)))
                na_ref.append(sum(1 for it in ITEMS if r[it] is None))
                na_pert.append(sum(1 for it in ITEMS if q[it] is None))
    kappa = cohen_kappa(kp)
    print(f"------------- PERTURBATION D (pooled over {runs} runs) -----------")
    if unpaired:
        print(f"  WARNING: {unpaired} quant runs had no matching meta by rate/seed")
    print(f"Cohen's kappa (binary diagnosis, pooled): {kappa:.3f}" if kappa is not None else "kappa: n/a")
    if per_item_abs:
        print(f"mean per-item drift (jointly-scored items): {mean(per_item_abs):.3f}  (n items = {len(per_item_abs)})")
    if total_abs:
        print(f"mean summed-total drift (N/A={NA_IN_TOTAL}): {mean(total_abs):.3f}")
    if na_ref:
        print(f"mean N/A items per run: ref {mean(na_ref):.2f} -> pert {mean(na_pert):.2f}")
