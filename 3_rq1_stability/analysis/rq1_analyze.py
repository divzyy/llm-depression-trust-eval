#!/usr/bin/env python3

import os, re, glob, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))
warnings.filterwarnings("ignore")

# CONFIG
ROOT = "analysis_output"
REFERENCE_BASELINE = os.path.join(ROOT, "Baseline")   
GT_DIR   = f"{DAIC}/labels"          
GT_CSVS  = [os.path.join(GT_DIR, "train_split_Depression_AVEC2017.csv"),
            os.path.join(GT_DIR, "dev_split_Depression_AVEC2017.csv")]
OUT_DIR  = os.path.join(ROOT, "rq1_analysis")                      
EXPECTED_BASE_FP = (41, 137, 0.587)   

PHQ8_THRESHOLD = 10
EXPECTED_TEST_SPLIT = (14, 27)   

PHQ8 = ["PHQ8_NoInterest","PHQ8_Depressed","PHQ8_Sleep","PHQ8_Tired",
        "PHQ8_Appetite","PHQ8_Failure","PHQ8_Concentrating","PHQ8_Moving"]
FIG_DIR = os.path.join(OUT_DIR, "figures"); TAB_DIR = os.path.join(OUT_DIR, "tables")

def _zero_shot_only(paths):
    """Drop any few-shot file (fs / fewshot / few_shot) so we never mix paths."""
    out = []
    for p in paths:
        b = os.path.basename(p).lower()
        if re.search(r'(fewshot|few_shot|few|_fs_|(^|_)fs[_.])', b):
            continue
        out.append(p)
    return out

def find_meta(run_dir):
    
    cands = []
    for pat in ["meta_review_zeroshot*.csv", "meta_review_zs_*.csv", "meta_review_zs*.csv"]:
        cands += glob.glob(os.path.join(run_dir, "**", pat), recursive=True)
    if not cands:
        cands = glob.glob(os.path.join(run_dir, "**", "meta", "*.csv"), recursive=True)
    cands = _zero_shot_only(cands)
    return sorted(set(cands))[0] if cands else None

def find_quant(run_dir):
    """Zero-shot quant item scores. Handles 'zero_shot'/'zs' names, falls back to any
    results csv inside a quan_zero_shot/ folder, and skips few-shot files."""
    cands = []
    for pat in ["results_zero_shot*.csv", "results_zs_*.csv", "results_zs*.csv"]:
        cands += glob.glob(os.path.join(run_dir, "**", pat), recursive=True)
    if not cands:
        c = glob.glob(os.path.join(run_dir, "**", "quan_zero_shot", "*.csv"), recursive=True)
        cands = [p for p in c if os.path.basename(p).lower().startswith("results")] or c
    cands = _zero_shot_only(cands)
    return sorted(set(cands))[0] if cands else None

def sev_to_bin(s): return 1 if s >= 2 else 0

def load_bin(meta_csv):
    """pid -> binary diagnosis from meta severity."""
    if not meta_csv or not os.path.exists(meta_csv): return {}
    df = pd.read_csv(meta_csv)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
    df = df.dropna(subset=["severity"])
    return {int(r.participant_id): sev_to_bin(int(r.severity)) for r in df.itertuples()}

def load_items(quant_csv):
    """pid -> {item: float or np.nan}."""
    if not quant_csv or not os.path.exists(quant_csv): return {}
    df = pd.read_csv(quant_csv)
    for it in PHQ8:
        df[it] = pd.to_numeric(df[it], errors="coerce") if it in df else np.nan
    return {int(r["participant_id"]): {it: r.get(it, np.nan) for it in PHQ8}
            for _, r in df.iterrows()}

def load_gt():
    
    missing = [p for p in GT_CSVS if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "AVEC2017 label CSVs not found: " + ", ".join(missing) +
            "\nSet GT_DIR. No fallback is used, because an exported ground-truth "
            "file may still carry the old PHQ8_Binary rule.")
    gt = pd.concat([pd.read_csv(p) for p in GT_CSVS], ignore_index=True)
    gt["PHQ8_Score"] = pd.to_numeric(gt["PHQ8_Score"], errors="coerce")
    lab = {int(r.Participant_ID): int(r.PHQ8_Score >= PHQ8_THRESHOLD)
           for r in gt.itertuples() if not np.isnan(r.PHQ8_Score)}
    
    if "PHQ8_Binary" in gt.columns:
        off = pd.to_numeric(gt["PHQ8_Binary"], errors="coerce")
        dis = gt[(gt["PHQ8_Score"] >= PHQ8_THRESHOLD).astype(int) != off]
        for r in dis.itertuples():
            print(f"  [gt audit] subject {int(r.Participant_ID)}: score={int(r.PHQ8_Score)} "
                  f"official PHQ8_Binary={int(r.PHQ8_Binary)} -> counted as "
                  f"{int(r.PHQ8_Score >= PHQ8_THRESHOLD)} by the >= {PHQ8_THRESHOLD} rule")
    return lab, f"AVEC2017:PHQ8_Score>={PHQ8_THRESHOLD}"

def run_stats(run_meta, run_quant):
    rb, rq = load_bin(run_meta), load_items(run_quant)
    pids = sorted(set(rb) & set(BASE_BIN) & set(GT))
    flips = cc = cw = wc = ww = 0
    b_seq, r_seq = [], []
    cw_ids, wc_ids = set(), set()
    for p in pids:
        b, r, g = BASE_BIN[p], rb[p], GT[p]
        b_seq.append(b); r_seq.append(r)
        if b != r: flips += 1
        bc, rc = (b == g), (r == g)
        if   bc and rc: cc += 1
        elif bc and not rc: cw += 1; cw_ids.add(p)
        elif not bc and rc: wc += 1; wc_ids.add(p)
        else: ww += 1
    n = len(pids)
    try: kappa = cohen_kappa_score(b_seq, r_seq) if n else np.nan
    except Exception: kappa = np.nan
    # drift vs baseline
    qp = sorted(set(rq) & set(BASE_Q))
    item_abs, item_signed, sum_abs, sum_signed, na_b, na_r = [], [], [], [], [], []
    for p in qp:
        bvals, rvals = BASE_Q[p], rq[p]
        na_b.append(sum(np.isnan(bvals[i]) for i in PHQ8))
        na_r.append(sum(np.isnan(rvals[i]) for i in PHQ8))
        for it in PHQ8:                                   
            if not np.isnan(bvals[it]) and not np.isnan(rvals[it]):
                item_abs.append(abs(rvals[it]-bvals[it])); item_signed.append(rvals[it]-bvals[it])
        bs = np.nansum([bvals[i] for i in PHQ8]); rs = np.nansum([rvals[i] for i in PHQ8])
        sum_abs.append(abs(rs-bs)); sum_signed.append(rs-bs)
    return dict(n=n, flips=flips, flip_rate=(flips/n if n else np.nan),
                consistency=(1-flips/n if n else np.nan), kappa=kappa,
                cc=cc, cw=cw, wc=wc, ww=ww, cw_ids=cw_ids, wc_ids=wc_ids,
                drift_item_mae=(np.mean(item_abs) if item_abs else np.nan),
                drift_item_signed=(np.mean(item_signed) if item_signed else np.nan),
                n_item_pairs=len(item_abs),
                drift_sum_abs=(np.mean(sum_abs) if sum_abs else np.nan),
                drift_sum_signed=(np.mean(sum_signed) if sum_signed else np.nan),
                na_base=(np.mean(na_b) if na_b else np.nan),
                na_run=(np.mean(na_r) if na_r else np.nan))

def row(name, s):
    return {"perturbation": name, "n": s["n"], "flip_rate": s["flip_rate"],
            "consistency": s["consistency"], "kappa": s["kappa"],
            "cc": s["cc"], "cw": s["cw"], "wc": s["wc"], "ww": s["ww"],
            "drift_item_mae": s["drift_item_mae"], "drift_item_signed": s["drift_item_signed"],
            "n_item_pairs": s["n_item_pairs"], "drift_sum_abs": s["drift_sum_abs"],
            "drift_sum_signed": s["drift_sum_signed"], "na_base": s["na_base"], "na_run": s["na_run"]}

def save_tab(df, name):
    df.to_csv(os.path.join(TAB_DIR, name+".csv"), index=False)
    open(os.path.join(TAB_DIR, name+".md"), "w").write(df.to_markdown(index=False, floatfmt=".3f"))
    print(f"  [table]  tables/{name}.csv/.md")

def save_fig(fig, name):
    for ext in ("pdf", "png"): fig.savefig(os.path.join(FIG_DIR, name+"."+ext), bbox_inches="tight", dpi=200)
    plt.close(fig); print(f"  [figure] figures/{name}.pdf/.png")

os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)
print("="*74); print(f"Reference baseline: {REFERENCE_BASELINE}"); print("="*74)
BASE_META, BASE_QUANT = find_meta(REFERENCE_BASELINE), find_quant(REFERENCE_BASELINE)
print(f"  meta : {BASE_META}\n  quant: {BASE_QUANT}")
BASE_BIN, BASE_Q = load_bin(BASE_META), load_items(BASE_QUANT)
_bq = pd.read_csv(BASE_QUANT)
_fp = (len(_bq), int(pd.to_numeric(_bq['num_questions_na'],errors='coerce').sum()),
       round(pd.to_numeric(_bq['avg_difference'],errors='coerce').mean(),3))
print(f"  fingerprint (n, total_NA, mean_avg_diff) = {_fp}   expected {EXPECTED_BASE_FP}")
if _fp != EXPECTED_BASE_FP: print("  *** WARNING: baseline fingerprint differs from the known reference — CONFIRM before trusting numbers.")
GT, gtsrc = load_gt(); print(f"  ground-truth source: {gtsrc}  (n={len(GT)})")

_test_pids = sorted(set(BASE_BIN) & set(GT))
_npos = sum(GT[p] for p in _test_pids); _nneg = len(_test_pids) - _npos
print(f"  test-set class split: {_npos} depressed / {_nneg} not depressed  "
      f"(expected {EXPECTED_TEST_SPLIT[0]} / {EXPECTED_TEST_SPLIT[1]})")
if (_npos, _nneg) != EXPECTED_TEST_SPLIT:
    print("  *** WARNING: class split differs from the expected 14/27 — CONFIRM before trusting the flip matrix.")

_base_wrong = [p for p in _test_pids if BASE_BIN[p] != GT[p]]
_fp_ids = [p for p in _base_wrong if GT[p] == 0]; _fn_ids = [p for p in _base_wrong if GT[p] == 1]
print(f"  reference run: {len(_test_pids)-len(_base_wrong)} correct / {len(_base_wrong)} wrong "
      f"({len(_fp_ids)} FP {sorted(_fp_ids)}, {len(_fn_ids)} FN {sorted(_fn_ids)})")

rows = []


vA = os.path.join(ROOT, "VariantA"); A_cw, A_wc = set(), set()
if os.path.isdir(vA):
    print("\n[VariantA] per-condition")
    for cond in sorted(os.listdir(vA)):
        cb = os.path.join(vA, cond)
        if not os.path.isdir(cb): continue
        m, q = find_meta(cb), find_quant(cb)
        if not m or not q:
            print(f"  {cond:6s} *** MISSING  meta={os.path.basename(m) if m else '-'} quant={os.path.basename(q) if q else '-'}")
            continue
        s = run_stats(m, q); rows.append(row(f"A/{cond}", s)); A_cw |= s["cw_ids"]; A_wc |= s["wc_ids"]
        print(f"  {cond:6s} [{os.path.basename(m)}] flip={s['flip_rate']:.3f} kappa={s['kappa']:.3f} cw={s['cw']} wc={s['wc']}")

for V in ["VariantB", "VariantC"]:
    vb = os.path.join(ROOT, V)
    if not os.path.isdir(vb): print(f"\n[{V}] folder not found"); continue
    m, q = find_meta(vb), find_quant(vb)
    if not m or not q: print(f"\n[{V}] *** MISSING meta={m} quant={q}"); continue
    s = run_stats(m, q); rows.append(row(V[-1], s))   # 'B' / 'C'
    print(f"\n[{V}] meta={m}\n       flip={s['flip_rate']:.3f} kappa={s['kappa']:.3f} "
          f"cw={s['cw']} wc={s['wc']} drift_item={s['drift_item_mae']:.3f} drift_sum={s['drift_sum_abs']:.3f}")


vD = os.path.join(ROOT, "VariantD"); D_rows = []; D_cw = D_wc = 0
D_cw_ids, D_wc_ids = {}, {}          
if os.path.isdir(vD):
    print("\n[VariantD] rate x seed")
    for rate in sorted((d for d in os.listdir(vD) if d.startswith("rate_")),
                       key=lambda x: int(x.split("_")[1])):
        rb = os.path.join(vD, rate); per_seed = []
        for seed in sorted((d for d in os.listdir(rb) if d.startswith("seed_")),
                           key=lambda x: int(x.split("_")[1])):
            sb = os.path.join(rb, seed); m, q = find_meta(sb), find_quant(sb)
            if not m or not q: print(f"  {rate}/{seed} *** MISSING"); continue
            s = run_stats(m, q); per_seed.append(s); D_cw += s["cw"]; D_wc += s["wc"]
            for p in s["cw_ids"]: D_cw_ids[p] = D_cw_ids.get(p, 0) + 1
            for p in s["wc_ids"]: D_wc_ids[p] = D_wc_ids.get(p, 0) + 1
        if not per_seed: continue
        fr = [x["flip_rate"] for x in per_seed]; di = [x["drift_item_mae"] for x in per_seed]
        ds = [x["drift_sum_abs"] for x in per_seed]
        kp = [x["kappa"] for x in per_seed]
        rec = {"rate_pct": int(rate.split("_")[1]), "n_seeds": len(per_seed),
               "flip_rate_mean": np.nanmean(fr), "flip_rate_std": np.nanstd(fr),
               "kappa_mean": np.nanmean(kp), "kappa_std": np.nanstd(kp),
               "drift_item_mean": np.nanmean(di), "drift_item_std": np.nanstd(di),
               "drift_sum_mean": np.nanmean(ds),  "drift_sum_std": np.nanstd(ds)}
        D_rows.append(rec)
        print(f"  rate_{rec['rate_pct']:>2}: seeds={rec['n_seeds']} "
              f"flip={rec['flip_rate_mean']:.3f}+/-{rec['flip_rate_std']:.3f} "
              f"drift_item={rec['drift_item_mean']:.3f} drift_sum={rec['drift_sum_mean']:.3f}")

print("\n--- writing tables ---")
if rows:   save_tab(pd.DataFrame(rows), "rq1_per_run_metrics")
if D_rows: save_tab(pd.DataFrame(D_rows), "rq1_D_dose_response")
flipdir = pd.DataFrame([{"perturbation":"A (unique subjects)","correct_to_wrong":len(A_cw),"wrong_to_correct":len(A_wc)}])
for lab in ["B","C"]:
    r = next((x for x in rows if x["perturbation"]==lab), None)
    if r: flipdir = pd.concat([flipdir, pd.DataFrame([{"perturbation":lab,"correct_to_wrong":r["cw"],"wrong_to_correct":r["wc"]}])], ignore_index=True)
flipdir = pd.concat([flipdir, pd.DataFrame([{"perturbation":"D (pooled seed-instances)","correct_to_wrong":D_cw,"wrong_to_correct":D_wc}])], ignore_index=True)
save_tab(flipdir, "rq1_flip_direction")


if D_cw_ids or D_wc_ids:
    det = []
    for p in sorted(set(D_cw_ids) | set(D_wc_ids)):
        det.append({"participant_id": p,
                    "gt_binary": GT.get(p, np.nan),
                    "reference_correct": int(BASE_BIN.get(p) == GT.get(p)),
                    "runs_correct_to_wrong": D_cw_ids.get(p, 0),
                    "runs_wrong_to_correct": D_wc_ids.get(p, 0)})
    det = pd.DataFrame(det)
    save_tab(det, "rq1_D_break_fix_by_participant")
    print(f"  [D break/fix] {D_cw} runs break a correct diagnosis, "
          f"{D_wc} runs fix a wrong one "
          f"(fixers: {sorted(D_wc_ids)})")

print("writing figures")
try:
    fig, ax = plt.subplots(figsize=(7,3.8))
    fd = flipdir.set_index("perturbation")[["correct_to_wrong","wrong_to_correct"]]
    fd.plot(kind="bar", ax=ax, color=["#c0392b","#2e86c1"])
    ax.set_ylabel("count"); ax.set_title("Flip direction (correct->wrong is destructive)")
    plt.xticks(rotation=20, ha="right"); save_fig(fig, "rq1_flip_direction")
except Exception as e: print(f"  [skip flip_direction] {e}")
for lab in ["B","C"]:
    r = next((x for x in rows if x["perturbation"]==lab), None)
    if not r: continue
    try:
        mat = np.array([[r["cc"], r["cw"]],[r["wc"], r["ww"]]])
        fig, ax = plt.subplots(figsize=(3.6,3.2))
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["stayed correct","became wrong"],
                    yticklabels=["was correct","was wrong"], ax=ax)
        ax.set_title(f"Perturbation {lab}: flip matrix"); save_fig(fig, f"rq1_flip_matrix_{lab}")
    except Exception as e: print(f"  [skip flip_matrix_{lab}] {e}")

if D_rows:
    try:
        d = pd.DataFrame(D_rows).sort_values("rate_pct")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9,3.6))
        a1.errorbar(d["rate_pct"], d["flip_rate_mean"], yerr=d["flip_rate_std"], marker="o", capsize=3)
        a1.set_xlabel("replacement rate (%)"); a1.set_ylabel("flip rate"); a1.set_title("D: diagnosis flip vs rate")
        a2.errorbar(d["rate_pct"], d["drift_item_mean"], yerr=d["drift_item_std"], marker="s", color="#7f6000", capsize=3)
        a2.set_xlabel("replacement rate (%)"); a2.set_ylabel("per-item score drift"); a2.set_title("D: score drift vs rate")
        fig.tight_layout(); save_fig(fig, "rq1_D_dose_response")
    except Exception as e: print(f"  [skip D_dose_response] {e}")

print(f"\nDone. Outputs in {os.path.abspath(OUT_DIR)}  (figures/ and tables/).")
