#!/usr/bin/env python3

import csv, re, os

REF_META  = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/qual/meta_review_zeroshot_test_v2.csv")
REF_QUANT = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/quan/results_zero_shot_test41.csv")
C_META    = os.path.expanduser("~/ai-psychiatrist/analysis_output/VariantC/meta/meta_review_zeroshot_varC.csv")
C_QUANT   = os.path.expanduser("~/ai-psychiatrist/analysis_output/VariantC/quan/results_zero_shot_varC.csv")

def ids_meta(path):
    s=set()
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            m=re.search(r'\d+', str(row.get('participant_id','')))
            sev=row.get('severity','')
            sv=re.search(r'-?\d+', str(sev))
            if m and sv: s.add(int(m.group()))   # only rows with a real severity
    return s

def ids_quant(path):
    s=set()
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            m=re.search(r'\d+', str(row.get('participant_id','')))
            if m: s.add(int(m.group()))
    return s

ref_m = ids_meta(REF_META)
ref_q = ids_quant(REF_QUANT)
c_m   = ids_meta(C_META)
c_q   = ids_quant(C_QUANT)

print(f"reference META ids: {len(ref_m)}")
print(f"reference QUANT ids: {len(ref_q)}")
print(f"VariantC META ids (with severity): {len(c_m)}")
print(f"VariantC QUANT ids: {len(c_q)}")
print()

flip_set = ref_m & c_m
print(f"FLIP script participant set (ref_meta & C_meta): {len(flip_set)}")
print(f"   dropped from reference: {sorted(ref_m - c_m)}")
print()

kappa_set = ref_m & c_m
drift_set = ref_q & c_q
print(f"KAPPA participant set (ref_meta & C_meta): {len(kappa_set)}")
print(f"DRIFT participant set (ref_quant & C_quant): {len(drift_set)}")
print(f"   dropped from reference quant: {sorted(ref_q - c_q)}")
print()

print("=== THE DIFFERENCE ===")
print(f"in FLIP set but not DRIFT set: {sorted(flip_set - drift_set)}")
print(f"in DRIFT set but not FLIP set: {sorted(drift_set - flip_set)}")
print(f"in KAPPA set but not FLIP set: {sorted(kappa_set - flip_set)}")
print()
print("If KAPPA set == FLIP set, the '38' in the kappa script was the QUANT (drift) count,")
print("not the meta (kappa) count. Check which number the kappa script actually printed as")
print("'meta pairs' vs 'quant common' for C.")
