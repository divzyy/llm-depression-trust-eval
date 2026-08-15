#!/usr/bin/env python3


import os, glob, csv, re, sys


D_ROOT      = os.path.expanduser("~/ai-psychiatrist/analysis_output/VariantD")
REF_CSV     = os.path.expanduser("~/ai-psychiatrist/analysis_output/Baseline/qual/meta_review_zeroshot_test_v2.csv")
TRAIN_LABELS= os.path.expanduser("~/daic_woz_data/labels/train_split_Depression_AVEC2017.csv")
DEV_LABELS  = os.path.expanduser("~/daic_woz_data/labels/dev_split_Depression_AVEC2017.csv")
META_FILENAME = "meta_review_zeroshot_test_v2.csv"   
DEPRESSED_CUT = 2                        
OUT_CSV     = os.path.expanduser("~/perturbation_d_flip_evidence.csv")

def to_int(x):
    if x is None: return None
    m = re.search(r"-?\d+", str(x))
    return int(m.group()) if m else None

def read_meta(path):
    """participant_id -> severity(int) from a meta_review CSV."""
    out = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            pid = to_int(row.get("participant_id"))
            sev = to_int(row.get("severity"))
            if pid is not None and sev is not None:
                out[pid] = sev
    return out

def read_labels(*paths):
   
    gt = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"WARNING: label file missing: {p}", file=sys.stderr); continue
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                
                pid = None; lab = None
                for k, v in row.items():
                    kl = k.lower().strip()
                    if kl in ("participant_id", "participant"): pid = to_int(v)
                    if "binary" in kl: lab = to_int(v)
                if pid is not None and lab is not None:
                    gt[pid] = lab
    return gt

def binary(sev):
    return 1 if sev >= DEPRESSED_CUT else 0

if not os.path.exists(REF_CSV):
    sys.exit(f"ERROR: reference run not found at {REF_CSV}")
ref = read_meta(REF_CSV)
gt  = read_labels(TRAIN_LABELS, DEV_LABELS)
print(f"Reference run: {len(ref)} participants")
print(f"Ground-truth labels loaded: {len(gt)} participants")
missing_gt = sorted(p for p in ref if p not in gt)
if missing_gt:
    print(f"WARNING: {len(missing_gt)} reference participants have no GT label: {missing_gt}")


run_files = sorted(glob.glob(os.path.join(D_ROOT, "rate_*", "seed_*", "meta", META_FILENAME)))
if not run_files:
    sys.exit(f"ERROR: no '{META_FILENAME}' found under {D_ROOT}/rate_*/seed_*/ . "
             f"Check D_ROOT and META_FILENAME.")

flips = harmful = helpful = comparisons = 0
runs_seen = 0
missing_in_run = 0
evidence = []

for path in run_files:
    m = re.search(r"rate_([^/]+)[/\\]seed_([^/]+)[/\\]", path)
    rate = m.group(1) if m else "?"
    seed = m.group(2) if m else "?"
    runs_seen += 1
    pert = read_meta(path)
    for pid, ref_sev in ref.items():
        if pid not in pert:           
            missing_in_run += 1
            continue
        comparisons += 1
        ref_bin  = binary(ref_sev)
        pert_bin = binary(pert[pid])
        if pert_bin != ref_bin:
            flips += 1
            direction = "unknown_no_GT"
            if pid in gt:
                ref_correct = (ref_bin == gt[pid])
                if ref_correct:
                    harmful += 1; direction = "harmful(correct->wrong)"
                else:
                    helpful += 1; direction = "helpful(wrong->correct)"
            evidence.append({
                "participant_id": pid, "rate": rate, "seed": seed,
                "ref_severity": ref_sev, "pert_severity": pert[pid],
                "ref_binary": ref_bin, "pert_binary": pert_bin,
                "gt_binary": gt.get(pid, ""), "direction": direction,
            })

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["participant_id","rate","seed",
        "ref_severity","pert_severity","ref_binary","pert_binary",
        "gt_binary","direction"])
    w.writeheader()
    for r in sorted(evidence, key=lambda r:(str(r["rate"]),str(r["seed"]),r["participant_id"])):
        w.writerow(r)

print("\nPERTURBATION D FLIP SUMMARY")
print(f"Rate/seed runs found : {runs_seen}")
print(f"Diagnosis comparisons: {comparisons}")
print(f"Participant-missing-in-run skips: {missing_in_run}")
print(f"TOTAL FLIPS (vs reference run): {flips}")
print(f"   harmful (correct -> wrong) : {harmful}")
print(f"   helpful (wrong -> correct) : {helpful}")
print(f"   flips with no GT label     : {flips - harmful - helpful}")
print(f"\nPer-flip evidence written to: {OUT_CSV}")
print("Draft currently claims: 68 flips, 64 harmful, 4 helpful.")
print("Compare the three numbers above against that claim.")


print("\nFlips by rate:")
by_rate = {}
for r in evidence:
    by_rate.setdefault(str(r["rate"]), [0,0,0])
    by_rate[str(r["rate"])][0] += 1
    if r["direction"].startswith("harmful"): by_rate[str(r["rate"])][1] += 1
    if r["direction"].startswith("helpful"): by_rate[str(r["rate"])][2] += 1
for rate in sorted(by_rate):
    t,h,he = by_rate[rate]
    print(f"   rate {rate}: {t} flips ({h} harmful, {he} helpful)")
