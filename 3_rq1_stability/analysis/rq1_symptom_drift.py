#!/usr/bin/env python3

import os, glob, re
import numpy as np, pandas as pd

ROOT = "analysis_output"
BASELINE = os.path.join(ROOT, "Baseline")     
VARIANT_D = os.path.join(ROOT, "VariantD")
OUT_DIR  = os.path.join(ROOT, "rq1_analysis", "tables")
PHQ8 = ["PHQ8_NoInterest","PHQ8_Depressed","PHQ8_Sleep","PHQ8_Tired",
        "PHQ8_Appetite","PHQ8_Failure","PHQ8_Concentrating","PHQ8_Moving"]
NICE = {"PHQ8_NoInterest":"No interest","PHQ8_Depressed":"Depressed mood",
        "PHQ8_Sleep":"Sleep","PHQ8_Tired":"Fatigue","PHQ8_Appetite":"Appetite",
        "PHQ8_Failure":"Failure","PHQ8_Concentrating":"Concentration","PHQ8_Moving":"Movement"}

def _zero_shot_only(paths):
    out = []
    for p in paths:
        b = os.path.basename(p).lower()
        if re.search(r'(fewshot|few_shot|few|_fs_|(^|_)fs[_.])', b):  # never a few-shot file
            continue
        out.append(p)
    return out

def find_quant(run_dir):
    cands = []
    for pat in ["results_zero_shot*.csv", "results_zs_*.csv", "results_zs*.csv"]:
        cands += glob.glob(os.path.join(run_dir, "**", pat), recursive=True)
    if not cands:
        c = glob.glob(os.path.join(run_dir, "**", "quan_zero_shot", "*.csv"), recursive=True)
        cands = [p for p in c if os.path.basename(p).lower().startswith("results")] or c
    cands = _zero_shot_only(cands)
    return sorted(set(cands))[0] if cands else None

def load_items(quant_csv):
    """pid -> {item: float or np.nan}. 'N/A' becomes NaN."""
    if not quant_csv or not os.path.exists(quant_csv): return {}
    df = pd.read_csv(quant_csv)
    for it in PHQ8:
        df[it] = pd.to_numeric(df[it], errors="coerce") if it in df.columns else np.nan
    return {int(r["participant_id"]): {it: r.get(it, np.nan) for it in PHQ8}
            for _, r in df.iterrows()}

os.makedirs(OUT_DIR, exist_ok=True)
BASE_Q = load_items(find_quant(BASELINE))
print(f"baseline quant: {find_quant(BASELINE)}  ({len(BASE_Q)} participants)")
if not BASE_Q: raise SystemExit("baseline quant not found -- check BASELINE path.")

records = []   
rates = sorted((d for d in os.listdir(VARIANT_D) if d.startswith("rate_")),
               key=lambda x: int(x.split("_")[1])) if os.path.isdir(VARIANT_D) else []
for rate in rates:
    rp = os.path.join(VARIANT_D, rate); rate_pct = int(rate.split("_")[1])
    for seed in sorted((d for d in os.listdir(rp) if d.startswith("seed_")),
                       key=lambda x: int(x.split("_")[1])):
        q = find_quant(os.path.join(rp, seed))
        if not q: print(f"  {rate}/{seed}: MISSING quant"); continue
        pert = load_items(q)
        for pid in set(pert) & set(BASE_Q):
            for it in PHQ8:
                b, r = BASE_Q[pid][it], pert[pid][it]
                if not np.isnan(b) and not np.isnan(r):     
                    records.append({"symptom": it, "rate": rate_pct,
                                    "abs": abs(r-b), "signed": r-b})
df = pd.DataFrame(records)
print(f"total jointly-scored item comparisons: {len(df)}")

rows = []
for it in PHQ8:
    s = df[df.symptom==it]
    rows.append({"Symptom": NICE[it], "n_comparisons": len(s),
                 "mean_abs_drift": s["abs"].mean() if len(s) else np.nan,
                 "mean_signed_drift": s["signed"].mean() if len(s) else np.nan,
                 "pct_changed": (s["abs"]>0).mean()*100 if len(s) else np.nan})
t1 = pd.DataFrame(rows)
t1.to_csv(os.path.join(OUT_DIR,"rq1_D_symptom_drift.csv"), index=False)
open(os.path.join(OUT_DIR,"rq1_D_symptom_drift.md"),"w").write(t1.to_markdown(index=False, floatfmt=".3f"))
print("\n=== per-symptom drift (Perturbation D, pooled over rates and seeds) ===")
print(t1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


piv = df.pivot_table(index="symptom", columns="rate", values="abs", aggfunc="mean").reindex(PHQ8)
piv.index = [NICE[i] for i in piv.index]
piv.to_csv(os.path.join(OUT_DIR,"rq1_D_symptom_drift_by_rate.csv"))
open(os.path.join(OUT_DIR,"rq1_D_symptom_drift_by_rate.md"),"w").write(piv.to_markdown(floatfmt=".3f"))
print("\n=== per-symptom mean absolute drift by rate (%) ===")
print(piv.to_string(float_format=lambda x: f"{x:.3f}"))


print("\ncheck -- overall mean abs drift by rate (should match tab:rq1-d 'drift per item'):")
print(df.groupby("rate")["abs"].mean().round(3).to_string())
print(f"\nWrote 2 tables to {os.path.abspath(OUT_DIR)}")
