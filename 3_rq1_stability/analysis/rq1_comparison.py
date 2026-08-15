#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

REPO_ROOT = f"{ROOT}"
OUTPUT_DIR = os.path.join(REPO_ROOT, "rq1_stability_analysis", "output")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary")

for d in [TABLES_DIR, FIGURES_DIR, SUMMARY_DIR]:
    os.makedirs(d, exist_ok=True)

PHQ8_ITEMS = [
    "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
    "PHQ8_Appetite", "PHQ8_Failure", "PHQ8_Concentrating", "PHQ8_Moving"
]

GT_DIR = f"{DAIC}/labels"
GT_PATHS = [
    os.path.join(GT_DIR, "train_split_Depression_AVEC2017.csv"),
    os.path.join(GT_DIR, "dev_split_Depression_AVEC2017.csv")
]

def phq8_to_severity(s):
    return 0 if s <= 4 else 1 if s <= 9 else 2 if s <= 14 else 3 if s <= 19 else 4

def severity_to_diagnosis(sev):
    return 0 if sev <= 1 else 1

def ordinal_krippendorff_alpha(y1, y2):
    """Computes Krippendorff's Alpha for two raters with ordinal data."""
    y1 = np.array(y1, dtype=float)
    y2 = np.array(y2, dtype=float)
    n = len(y1)
    if n <= 1:
        return 1.0
    do = np.mean((y1 - y2) ** 2)
    combined = np.concatenate([y1, y2])
    var_combined = np.var(combined, ddof=1)
    de = 2 * var_combined
    if de == 0:
        return 1.0 if do == 0 else 0.0
    return 1.0 - (do / de)

def bootstrap_pfr_ci(y1, y2, n_bootstrap=2000, ci=0.95):
   
    y1 = np.array(y1)
    y2 = np.array(y2)
    flips = (y1 != y2).astype(int)
    n = len(flips)
    if n == 0:
        return 0.0, 0.0
    boot_pfrs = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n, size=n, replace=True)
        boot_pfrs.append(np.mean(flips[boot_idx]))
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_pfrs, alpha * 100)
    upper = np.percentile(boot_pfrs, (1 - alpha) * 100)
    return lower, upper

def run_mcnemar(y_true, y_pred1, y_pred2):
    
    b = sum((y_pred1 == y_true) & (y_pred2 != y_true))
    c = sum((y_pred1 != y_true) & (y_pred2 == y_true))
    
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_val = 2 * stats.binom.cdf(k, n, 0.5)
    return min(p_val, 1.0)

def jaccard_similarity(text1, text2):
    
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 and not words2:
        return 1.0
    return len(words1 & words2) / len(words1 | words2)

def unweighted_kappa(y1, y2):
   
    y1, y2 = np.array(y1, dtype=int), np.array(y2, dtype=int)
    if len(np.unique(np.concatenate([y1, y2]))) < 2:
        return np.nan
    try:
        return cohen_kappa_score(y1, y2)
    except Exception:
        return np.nan

def weighted_kappa(y1, y2):
    
    y1, y2 = np.array(y1, dtype=int), np.array(y2, dtype=int)
    if len(np.unique(np.concatenate([y1, y2]))) < 2:
        return np.nan
    try:
        return cohen_kappa_score(y1, y2, weights='linear')
    except Exception:
        return np.nan

def load_meta(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if "participant_id" in df.columns and "severity" in df.columns:
            df["participant_id"] = df["participant_id"].astype(int)
            df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
            return df.dropna(subset=["severity"]).set_index("participant_id")["severity"].to_dict()
    except Exception as e:
        print(f"Warning: Failed to load meta {filepath}: {e}")
    return None

def load_qualitative(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if "participant_id" in df.columns and "qualitative_assessment" in df.columns:
            df["participant_id"] = df["participant_id"].astype(int)
            return df.set_index("participant_id")["qualitative_assessment"].to_dict()
    except Exception as e:
        print(f"Warning: Failed to load qualitative {filepath}: {e}")
    return None

def load_quantitative(filepath):
    if not os.path.exists(filepath):
        return None
    out = {}
    try:
        if filepath.endswith(".jsonl"):
            with open(filepath) as f:
                for line in f:
                    if not line.strip(): continue
                    r = json.loads(line)
                    pid = r.get("participant_id")
                    if pid is None: continue
                    scores = {}
                    for it in PHQ8_ITEMS:
                        s = r.get(it, {}).get("score", None) if isinstance(r.get(it), dict) else r.get(it)
                        try:
                            scores[it] = float(s)
                        except (TypeError, ValueError):
                            scores[it] = np.nan
                    out[int(pid)] = scores
        else:
            df = pd.read_csv(filepath)
            df["participant_id"] = df["participant_id"].astype(int)
            for _, row in df.iterrows():
                pid = row["participant_id"]
                scores = {}
                for it in PHQ8_ITEMS:
                    s = row.get(it)
                    try:
                        scores[it] = float(s)
                    except (TypeError, ValueError):
                        scores[it] = np.nan
                out[pid] = scores
    except Exception as e:
        print(f"Warning: Failed to load quantitative {filepath}: {e}")
    return out if out else None

def load_gt():
    out = {}
    for p in GT_PATHS:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            pid_col = next((c for c in df.columns if c.lower() == "participant_id"), None)
            score_col = next((c for c in df.columns if c.lower() == "phq8_score"), None)
            if pid_col and score_col:
                for _, row in df.iterrows():
                    try:
                        out[int(row[pid_col])] = float(row[score_col])
                    except (TypeError, ValueError):
                        pass
        except Exception as e:
            print(f"Warning: Failed to load ground truth {p}: {e}")
    return out

def analyze_condition(cond):
    print(f"\n ---- \nAnalyzing Condition: {cond.upper()}\n -----")
    
    gt_scores = load_gt()
    if not gt_scores:
        print("Warning: Ground-truth files not found. Skipping accuracy validations.")
    gt_severity = {pid: phq8_to_severity(score) for pid, score in gt_scores.items()}
    gt_diagnosis = {pid: severity_to_diagnosis(sev) for pid, sev in gt_severity.items()}

    if cond == "few_shot":
        meta_filename = "meta_review_fewshot_test_v2.csv"
        quan_filename = "ids_test_chunk_8_step_2_dim_4096_examples_2_embedding_results_analysis_1.jsonl"
        quan_folder = "quan_gemma_few_shot"
    else:
        meta_filename = "meta_review_zeroshot_test_v2.csv"
        quan_filename = "results_zero_shot_test41.csv"
        quan_folder = "quan"

    runs = {
        "Baseline": {
            "meta": os.path.join(REPO_ROOT, "analysis_output", "Baseline", "qual", meta_filename),
            "qual": os.path.join(REPO_ROOT, "analysis_output", "Baseline", "qual", "qual_assessment_GEMMA_v2.csv"),
            "quan": os.path.join(REPO_ROOT, "analysis_output", "Baseline", quan_folder, quan_filename)
        },
        "New_Baseline": {
            "meta": os.path.join(REPO_ROOT, "analysis_output", "New_Baseline", "qual", meta_filename),
            "qual": os.path.join(REPO_ROOT, "analysis_output", "New_Baseline", "qual", "qual_assessment_GEMMA_v2.csv"),
            "quan": os.path.join(REPO_ROOT, "analysis_output", "New_Baseline", quan_folder, quan_filename)
        },
        "VariantB": {
            "meta": os.path.join(REPO_ROOT, "analysis_output", "VariantB", "meta", meta_filename),
            "qual": os.path.join(REPO_ROOT, "analysis_output", "VariantB", "qual", "qual_assessment_GEMMA_v2.csv"),
            "quan": os.path.join(REPO_ROOT, "analysis_output", "VariantB", "quan_" + cond, quan_filename)
        },
        "VariantC": {
            "meta": os.path.join(REPO_ROOT, "analysis_output", "VariantC", "meta", f"meta_review_{cond}_varC.csv"),
            "qual": os.path.join(REPO_ROOT, "analysis_output", "VariantC", "qual", "qual_assessment_varC.csv"),
            "quan": os.path.join(REPO_ROOT, "analysis_output", "VariantC", "quan_" + cond, quan_filename)
        }
    }

    if cond == "few_shot":
        runs["backup_baseline"] = {
            "meta": os.path.join(REPO_ROOT, "analysis_output", "backup_baseline_20260621_101507", "qual", "meta_review_fewshot_test_v2.csv"),
            "qual": os.path.join(REPO_ROOT, "analysis_output", "backup_baseline_20260621_101507", "qual", "qual_assessment_GEMMA_v2.csv"),
            "quan": None
        }

    
    rates = [5, 10, 20, 50]
    seeds = [1, 2, 3, 4, 5]
    for r in rates:
        for s in seeds:
            runs[f"VariantD_r{r}_s{s}"] = {
                "meta": os.path.join(REPO_ROOT, "analysis_output", "VariantD", f"rate_{r}", f"seed_{s}", "meta", meta_filename),
                "qual": os.path.join(REPO_ROOT, "analysis_output", "VariantD", f"rate_{r}", f"seed_{s}", "qual", "qual_assessment_GEMMA_v2.csv"),
                "quan": os.path.join(REPO_ROOT, "analysis_output", "VariantD", f"rate_{r}", f"seed_{s}", "quan_" + cond, quan_filename)
            }

    loaded_meta = {}
    loaded_qual = {}
    loaded_quan = {}

    for run_name, paths in runs.items():
        meta_data = load_meta(paths["meta"])
        qual_data = load_qualitative(paths["qual"])
        quan_data = load_quantitative(paths["quan"]) if paths.get("quan") else None
        
        if meta_data: loaded_meta[run_name] = meta_data
        if qual_data: loaded_qual[run_name] = qual_data
        if quan_data: loaded_quan[run_name] = quan_data

    if "Baseline" not in loaded_meta:
        print("ERROR: Primary Baseline meta-review output is missing. Cannot proceed.")
        return

    base_meta = loaded_meta["Baseline"]
    common_participants = sorted(list(base_meta.keys()))
    print(f"Loaded {len(common_participants)} common participants for evaluation.")

    records = []

    for run_name in loaded_meta.keys():
        if run_name == "Baseline":
            continue
            
        r_meta = loaded_meta[run_name]
        pids = [p for p in common_participants if p in r_meta]
        if len(pids) < len(common_participants):
            missing = set(common_participants) - set(r_meta.keys())
            print(f"Warning: {run_name} has missing participants: {list(missing)}")
            
        if not pids:
            continue

        y_base = np.array([base_meta[p] for p in pids])
        y_variant = np.array([r_meta[p] for p in pids])

        pfr = np.mean(y_base != y_variant)
        mad = np.mean(np.abs(y_base - y_variant))
        agreement = np.mean(y_base == y_variant)
        kappa = unweighted_kappa(y_base, y_variant)
        w_kappa = weighted_kappa(y_base, y_variant)
        k_alpha = ordinal_krippendorff_alpha(y_base, y_variant)
        
        if np.all(y_base == y_variant):
            wilcoxon_p = 1.0
        else:
            _, wilcoxon_p = stats.wilcoxon(y_base, y_variant)

        
        ci_lower, ci_upper = bootstrap_pfr_ci(y_base, y_variant)

        diag_base = np.array([severity_to_diagnosis(sev) for sev in y_base])
        diag_variant = np.array([severity_to_diagnosis(sev) for sev in y_variant])
        diag_pfr = np.mean(diag_base != diag_variant)

        r_qual = loaded_qual.get(run_name, {})
        base_qual = loaded_qual.get("Baseline", {})
        qual_sims = []
        for p in pids:
            if p in r_qual and p in base_qual:
                qual_sims.append(jaccard_similarity(base_qual[p], r_qual[p]))
        mean_qual_sim = np.mean(qual_sims) if qual_sims else np.nan

        r_quan = loaded_quan.get(run_name, {})
        base_quan = loaded_quan.get("Baseline", {})
        item_diffs_all = []         
        item_flips_all = []          
        pid_quan_agreed  = {}        
        pid_quan_n_flips = {}       

        if r_quan and base_quan:
            for p in pids:
                if p in r_quan and p in base_quan:
                    p_base_scores    = base_quan[p]
                    p_variant_scores = r_quan[p]
                    p_flips = 0
                    p_compared = 0
                    for it in PHQ8_ITEMS:
                        v_base    = p_base_scores.get(it, np.nan)
                        v_variant = p_variant_scores.get(it, np.nan)
                        if not (np.isnan(v_base) or np.isnan(v_variant)):
                            diff = abs(v_base - v_variant)
                            item_diffs_all.append(diff)
                            flipped = int(v_base != v_variant)
                            item_flips_all.append(flipped)
                            p_flips    += flipped
                            p_compared += 1
                    if p_compared > 0:
                        pid_quan_agreed[p]  = (p_flips == 0)
                        pid_quan_n_flips[p] = p_flips

        mean_quan_diff = np.mean(item_diffs_all) if item_diffs_all else np.nan
        mean_quan_flip = np.mean(item_flips_all) if item_flips_all else np.nan

        pids_quan_ok = [p for p in pids if p in pid_quan_agreed]

        if pids_quan_ok:
           
            pids_quan_agreed_list = [p for p in pids_quan_ok if pid_quan_agreed[p]]
            if pids_quan_agreed_list:
                meta_flips_given_quan_agreed = [
                    int(base_meta[p] != r_meta[p]) for p in pids_quan_agreed_list
                ]
                cond_pfr = np.mean(meta_flips_given_quan_agreed)  # pure meta instability
                cond_n   = len(pids_quan_agreed_list)
            else:
                cond_pfr, cond_n = np.nan, 0

            
            pids_meta_flipped = [p for p in pids_quan_ok if base_meta[p] != r_meta[p]]
            if pids_meta_flipped:
                had_prior_quan_flip = [
                    int(not pid_quan_agreed[p]) for p in pids_meta_flipped
                ]
                cascade_rate = np.mean(had_prior_quan_flip) 
                cascade_n    = len(pids_meta_flipped)
            else:
                cascade_rate, cascade_n = np.nan, 0
        else:
            cond_pfr = cond_n = cascade_rate = cascade_n = np.nan

        mae_base = np.nan
        mae_variant = np.nan
        acc_base = np.nan
        acc_variant = np.nan
        mcnemar_p = np.nan

        if gt_scores:
            y_gt   = np.array([gt_severity[p]  for p in pids])
            diag_gt = np.array([gt_diagnosis[p] for p in pids])

            mae_base    = np.mean(np.abs(y_base    - y_gt))
            mae_variant = np.mean(np.abs(y_variant - y_gt))
            acc_base    = np.mean(y_base    == y_gt)
            acc_variant = np.mean(y_variant == y_gt)

            mcnemar_p = run_mcnemar(diag_gt, diag_base, diag_variant)

        record = {
            "Run":                    run_name,
            "PFR":                    pfr,
            "PFR_CI_Lower":           ci_lower,
            "PFR_CI_Upper":           ci_upper,
            "MAD":                    mad,
            "Agreement":              agreement,
            "Kappa":                  kappa,
            "Weighted_Kappa":         w_kappa,
            "Krippendorff_Alpha":     k_alpha,
            "Wilcoxon_p":             wilcoxon_p,
            "Diag_PFR":               diag_pfr,
            "Qual_Similarity":        mean_qual_sim,
            "Quan_Item_MAD":          mean_quan_diff,
            "Quan_Item_PFR":          mean_quan_flip,
            "Cond_PFR_QuanAgreed":    cond_pfr,    # meta flip rate when quan agreed
            "Cond_N_QuanAgreed":      cond_n,      # n participants in that stratum
            "Cascade_Rate":           cascade_rate, # share of meta flips with prior quan flip
            "Cascade_N":              cascade_n,    # n meta-flip participants examined
            "MAE_Base":               mae_base,
            "MAE_Variant":            mae_variant,
            "Acc_Base":               acc_base,
            "Acc_Variant":            acc_variant,
            "McNemar_p":              mcnemar_p,
            "Count":                  len(pids)
        }
        records.append(record)

    df_results = pd.DataFrame(records)

    
    df_varD = df_results[df_results["Run"].str.startswith("VariantD")].copy()
    if not df_varD.empty:
        df_varD["Rate"] = df_varD["Run"].apply(lambda x: int(x.split("_")[1][1:]))
        df_varD["Seed"] = df_varD["Run"].apply(lambda x: int(x.split("_")[2][1:]))
        
        save_table(df_varD, f"detailed_VariantD_{cond}", TABLES_DIR)
        
        summary_varD = df_varD.groupby("Rate").agg({
            "PFR": ["mean", "std"],
            "MAD": ["mean", "std"],
            "Kappa": ["mean", "std"],
            "Weighted_Kappa": ["mean", "std"],
            "Krippendorff_Alpha": ["mean", "std"],
            "Qual_Similarity": ["mean", "std"],
            "Quan_Item_MAD": ["mean", "std"],
            "Quan_Item_PFR": ["mean", "std"],
            "MAE_Variant": ["mean", "std"],
            "Acc_Variant": ["mean", "std"],
            "Count": "sum"
        })
        summary_varD.columns = [f"{col[0]}_{col[1]}" for col in summary_varD.columns]
        summary_varD = summary_varD.reset_index()
        save_table(summary_varD, f"summary_VariantD_{cond}", TABLES_DIR)

        corr_coef, corr_p = stats.spearmanr(df_varD["Rate"], df_varD["PFR"])
        print(f"Variant D Spearman Trend (Rate vs PFR): coefficient={corr_coef:.4f}, p-value={corr_p:.4f}")
        with open(os.path.join(SUMMARY_DIR, f"variantD_trend_{cond}.txt"), "w") as f:
            f.write(f"Spearman rank correlation coefficient: {corr_coef:.6f}\n")
            f.write(f"Spearman p-value: {corr_p:.6e}\n")

    df_overall = df_results[~df_results["Run"].str.startswith("VariantD")].copy()
    
    
    if not df_varD.empty:
        agg_records = []
        for rate in rates:
            df_sub = df_varD[df_varD["Rate"] == rate]
            if df_sub.empty: continue
            agg_record = {
                "Run":                  f"VariantD_Rate{rate}_Aggregated",
                "PFR":                  df_sub["PFR"].mean(),
                "PFR_CI_Lower":         df_sub["PFR_CI_Lower"].mean(),
                "PFR_CI_Upper":         df_sub["PFR_CI_Upper"].mean(),
                "MAD":                  df_sub["MAD"].mean(),
                "Agreement":            df_sub["Agreement"].mean(),
                "Kappa":                df_sub["Kappa"].mean(),
                "Weighted_Kappa":       df_sub["Weighted_Kappa"].mean(),
                "Krippendorff_Alpha":   df_sub["Krippendorff_Alpha"].mean(),
                "Wilcoxon_p":           df_sub["Wilcoxon_p"].mean(),
                "Diag_PFR":             df_sub["Diag_PFR"].mean(),
                "Qual_Similarity":      df_sub["Qual_Similarity"].mean(),
                "Quan_Item_MAD":        df_sub["Quan_Item_MAD"].mean(),
                "Quan_Item_PFR":        df_sub["Quan_Item_PFR"].mean(),
                "Cond_PFR_QuanAgreed":  df_sub["Cond_PFR_QuanAgreed"].mean(),
                "Cond_N_QuanAgreed":    int(df_sub["Cond_N_QuanAgreed"].mean()),
                "Cascade_Rate":         df_sub["Cascade_Rate"].mean(),
                "Cascade_N":            int(df_sub["Cascade_N"].mean()),
                "MAE_Base":             df_sub["MAE_Base"].mean(),
                "MAE_Variant":          df_sub["MAE_Variant"].mean(),
                "Acc_Base":             df_sub["Acc_Base"].mean(),
                "Acc_Variant":          df_sub["Acc_Variant"].mean(),
                "McNemar_p":            df_sub["McNemar_p"].mean(),
                "Count":                int(df_sub["Count"].mean())
            }
            agg_records.append(agg_record)
        df_overall = pd.concat([df_overall, pd.DataFrame(agg_records)], ignore_index=True)

    save_table(df_overall, f"overall_stability_results_{cond}", TABLES_DIR)

    generate_plots(df_results, df_varD, cond)

    print("\n--- Run-to-Run Consistency ---")
    df_consistency = df_overall[df_overall["Run"].str.contains("Baseline")].copy()
    for _, row in df_consistency.iterrows():
        print(f"Comparison: Baseline vs {row['Run']}")
        print(f"  PFR (Severity):      {row['PFR']*100:.2f}% (95% CI: {row['PFR_CI_Lower']*100:.2f}%–{row['PFR_CI_Upper']*100:.2f}%)")
        print(f"  MAD:                 {row['MAD']:.4f}")
        print(f"  Weighted Kappa:      {row['Weighted_Kappa']:.4f}")
        print(f"  Krippendorff Alpha:  {row['Krippendorff_Alpha']:.4f}")
        print(f"  Wilcoxon p-value:    {row['Wilcoxon_p']:.4e}")

    
    print("\n--- Stage-Attribution Analysis ---")
    print("(Clinical meaning preservation defined behaviourally via PHQ-8 item score agreement)")
    attr_rows = []
    for _, row in df_overall.iterrows():
        if pd.isna(row.get("Cond_PFR_QuanAgreed")):
            continue
        attr_rows.append({
            "Run":                 row["Run"],
            "PFR_overall":         row["PFR"],
            "Quan_Item_PFR":       row["Quan_Item_PFR"],
            "Cond_PFR_QuanAgreed": row["Cond_PFR_QuanAgreed"],
            "Cond_N_QuanAgreed":   row["Cond_N_QuanAgreed"],
            "Cascade_Rate":        row["Cascade_Rate"],
            "Cascade_N":           row["Cascade_N"],
        })
        print(
            f"  {row['Run']:45s}  "
            f"PFR={row['PFR']*100:5.1f}%  "
            f"Cond_PFR(quan_ok)={row['Cond_PFR_QuanAgreed']*100:5.1f}% (n={int(row['Cond_N_QuanAgreed'])})  "
            f"Cascade={row['Cascade_Rate']*100:5.1f}% (n={int(row['Cascade_N'])})"
        )

    if attr_rows:
        df_attr = pd.DataFrame(attr_rows)
        save_table(df_attr, f"stage_attribution_{cond}", TABLES_DIR)
        with open(os.path.join(SUMMARY_DIR, f"stage_attribution_{cond}.txt"), "w") as f:
            f.write("Stage-Attribution Analysis\n")
            f.write("==========================\n")
            f.write("Clinical meaning preservation is defined behaviourally:\n")
            f.write("  - A participant\'s clinical content is considered preserved if the\n")
            f.write("    quantitative agent assigns identical PHQ-8 item scores on the\n")
            f.write("    perturbed transcript as on the original.\n\n")
            f.write("Metrics:\n")
            f.write("  Cond_PFR_QuanAgreed : meta-reviewer flip rate among participants where\n")
            f.write("                        clinical content was preserved (quan agreed).\n")
            f.write("                        = pure meta-reviewer instability.\n")
            f.write("  Cascade_Rate        : fraction of meta-reviewer flips that were\n")
            f.write("                        preceded by a quantitative agent flip.\n")
            f.write("                        = share of instability explained by upstream propagation.\n\n")
            f.write(df_attr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            f.write("\n")

def save_table(df, name, outdir):
    df.to_csv(os.path.join(outdir, name + ".csv"), index=False)
    with open(os.path.join(outdir, name + ".md"), "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"  [table] saved {name}.csv and {name}.md")

def generate_plots(df_results, df_varD, cond):
    # 1. Dose-Response Curve (Variant D)
    if not df_varD.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        df_plot = df_varD.groupby("Rate")["PFR"].agg(["mean", "std"]).reset_index()
        
        ax.errorbar(
            df_plot["Rate"], df_plot["mean"] * 100, yerr=df_plot["std"] * 100,
            fmt='-o', color='teal', linewidth=2.5, elinewidth=1.5, capsize=5, ecolor='darkgray', markersize=8
        )
        ax.set_xlabel("Lexical Synonym Replacement Rate (%)", fontweight='bold')
        ax.set_ylabel("Severity Level Prediction Flip Rate (%)", fontweight='bold')
        ax.set_title(f"Variant D Dose-Response Curve ({cond.replace('_', ' ').title()})", pad=15)
        ax.set_ylim(-5, 105)
        ax.set_xlim(-5, 55)
        ax.set_xticks([5, 10, 20, 50])
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_fig(fig, f"dose_response_curve_{cond}", FIGURES_DIR)

    
    main_variants = ["VariantB", "VariantC"]
    df_main = df_results[df_results["Run"].isin(main_variants)].copy()
    
    if not df_varD.empty:
        for rate in [5, 10, 20, 50]:
            df_sub = df_varD[df_varD["Rate"] == rate]
            if df_sub.empty: continue
            agg_row = pd.DataFrame([{
                "Run": f"VariantD_{rate}%",
                "PFR": df_sub["PFR"].mean(),
                "MAD": df_sub["MAD"].mean(),
                "Weighted_Kappa": df_sub["Weighted_Kappa"].mean(),
                "Count": int(df_sub["Count"].mean())
            }])
            df_main = pd.concat([df_main, agg_row], ignore_index=True)

    if not df_main.empty:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        color = 'steelblue'
        sns.barplot(data=df_main, x="Run", y=df_main["PFR"] * 100, ax=ax1, color=color, alpha=0.8)
        ax1.set_ylabel("Prediction Flip Rate (%)", color=color, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlabel("Perturbation Variant", fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15)
        
        ax2 = ax1.twinx()
        color = 'darkorange'
        ax2.plot(df_main["Run"], df_main["MAD"], color=color, marker='s', linewidth=2, markersize=8)
        ax2.set_ylabel("Mean Absolute Deviation (MAD)", color=color, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(False)
        
        plt.title(f"Stability Metrics Across Perturbation Variants ({cond.replace('_', ' ').title()})", pad=15)
        plt.tight_layout()
        save_fig(fig, f"variants_stability_comparison_{cond}", FIGURES_DIR)

    if not df_main.empty and "Acc_Base" in df_main.columns and not df_main["Acc_Base"].isna().all():
        fig, ax = plt.subplots(figsize=(8, 5))
        
        accs = [{"Run": "Baseline", "Accuracy": df_main["Acc_Base"].iloc[0] * 100}]
        for _, row in df_main.iterrows():
            accs.append({"Run": row["Run"], "Accuracy": row["Acc_Variant"] * 100})
        df_acc = pd.DataFrame(accs)
        
        sns.barplot(data=df_acc, x="Run", y="Accuracy", palette="Blues_d", alpha=0.9)
        ax.set_ylabel("Accuracy against Ground Truth (%)", fontweight='bold')
        ax.set_xlabel("Model Run", fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 2, f"{height:.1f}%", ha="center")
            
        plt.title(f"Depression Classification Accuracy vs. Ground-Truth Labels ({cond.replace('_', ' ').title()})", pad=15)
        plt.tight_layout()
        save_fig(fig, f"accuracy_performance_comparison_{cond}", FIGURES_DIR)

def save_fig(fig, name, outdir):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, name + "." + ext), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  [figure] saved {name}.pdf and {name}.png")

if __name__ == "__main__":
    for cond in ["zero_shot", "few_shot"]:
        analyze_condition(cond)
    print("\nRQ1 Stability Analysis Completed successfully!")
