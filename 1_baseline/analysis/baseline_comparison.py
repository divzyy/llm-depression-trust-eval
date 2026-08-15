#!/usr/bin/env python3
"""
baseline_comparison.py -- baseline / replication analysis.
  PART A -- vs GROUND TRUTH: is the reproduction any good?
  PART B -- vs the ORIGINAL PAPER: is the reproduction faithful?
Parts needing ground truth are skipped (with a notice) if GT CSVs are missing.
Deps: pandas numpy matplotlib seaborn scikit-learn tabulate

GROUND-TRUTH RULE (changed):
  The binary label is derived from the PHQ-8 total score with the >= 10
  screening cut-off, NOT from the official PHQ8_Binary column. For subject 409
  the official column reports 0 although the total is exactly 10. The original
  authors score with the >= 10 rule (their reported rates reconstruct only under
  14 depressed / 27 not on the 41-subject test split), so this keeps our numbers
  directly comparable with theirs. See thesis Section 4.2.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))
warnings.filterwarnings("ignore")

BASE_DIR = f"{ROOT}/analysis_output/Baseline"  # contains qual/ quan/ quan_gemma_few_shot/
GT_DIR   = f"{DAIC}/labels"             # AVEC2017 split CSVs

CONFIG = {
    "quant_zeroshot_csv":  os.path.join(BASE_DIR, "quan",
                                        "results_zero_shot_test41.csv"),
    "quant_fewshot_jsonl": os.path.join(BASE_DIR, "quan_gemma_few_shot",
                                        "ids_test_chunk_8_step_2_dim_4096_examples_2_embedding_results_analysis_1.jsonl"),
    "meta_zeroshot_csv":   os.path.join(BASE_DIR, "qual",
                                        "meta_review_zeroshot_test_v2.csv"),
    "meta_fewshot_csv":    os.path.join(BASE_DIR, "qual",
                                        "meta_review_fewshot_test_v2.csv"),
    "gt_csvs": [os.path.join(GT_DIR, "train_split_Depression_AVEC2017.csv"),
                os.path.join(GT_DIR, "dev_split_Depression_AVEC2017.csv")],
    "output_dir": "baseline_out",
}

# Original paper's reported results (paper 0: abstract + Sec 3.2-3.4 + Table 1).
# Table 1 rows: "Ground Truth vs Agent" and "Ground Truth vs Human".
PAPER_REPORTED = {
    "binary_accuracy":          0.780,
    "binary_balanced_accuracy": 0.730,
    "binary_precision":         0.727,
    "binary_recall":            0.571,
    "binary_f1":                0.640,
    # human expert (PhD-level researcher, guided by a senior clinical expert)
    "human_accuracy":           0.780,
    "human_balanced_accuracy":  0.679,
    "human_precision":          1.000,
    "human_recall":             0.357,
    "human_f1":                 0.526,
    "human_agent_agreement":    0.805,
    "quant_mae_zeroshot": 0.796,
    "quant_mae_fewshot":  0.619,
    "no_prediction_rate": 0.50,
}

PHQ8_ITEMS = ["PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
              "PHQ8_Appetite", "PHQ8_Failure", "PHQ8_Concentrating", "PHQ8_Moving"]

# conversion rules (from the original ai-psychiatrist repo)
def phq8_to_severity(s):
    return 0 if s <= 4 else 1 if s <= 9 else 2 if s <= 14 else 3 if s <= 19 else 4
def phq8_to_diagnosis(s):       # 0-9 = negative, >=10 = positive
    return 0 if s <= 9 else 1
def severity_to_diagnosis(sev): # severity 0-1 = negative, >=2 = positive
    return 0 if sev <= 1 else 1

def _exists(p): return bool(p) and os.path.exists(p)

def save_table(df, name, outdir):
    df.to_csv(os.path.join(outdir, name + ".csv"), index=False)
    with open(os.path.join(outdir, name + ".md"), "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
    print(f"  [table]  {name}.csv / .md")

def save_fig(fig, name, outdir):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, name + "." + ext), bbox_inches="tight", dpi=200)
    plt.close(fig); print(f"  [figure] {name}.pdf / .png")

def load_quant(source):
    rows = []
    if source.endswith(".jsonl"):
        with open(source) as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line); out = {"participant_id": r["participant_id"]}
                for it in PHQ8_ITEMS:
                    s = r.get(it, {}).get("score", None) if isinstance(r.get(it), dict) else r.get(it)
                    try: out[it] = float(s)
                    except (TypeError, ValueError): out[it] = np.nan
                rows.append(out)
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(source)[["participant_id"] + PHQ8_ITEMS].copy()
        for it in PHQ8_ITEMS:
            df[it] = pd.to_numeric(df[it], errors="coerce")   # "N/A" -> NaN
    df["n_answered"] = df[PHQ8_ITEMS].notna().sum(axis=1)
    return df

def load_meta(csv):
    df = pd.read_csv(csv)[["participant_id", "severity"]].copy()
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
    return df.dropna(subset=["severity"])

def load_gt(cfg):
    paths = cfg.get("gt_csvs") or []
    if not paths or not all(_exists(p) for p in paths): return None
    gt = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return gt.sort_values("Participant_ID").reset_index(drop=True)

def binary_metrics(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * prec * tpr / (prec + tpr) if prec and tpr and (prec + tpr) else np.nan
    return dict(accuracy=acc, balanced_accuracy=np.nanmean([tpr, tnr]),
                precision=prec, recall=tpr, f1=f1)

def wilson_ci(k, n, z=1.96):
    import math
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half, centre + half)

# label audit -- documents the 409 decision with real numbers
def label_audit(gt, cfg, outdir):
    """Report the class split under the >=10 rule and list every subject whose
    official PHQ8_Binary disagrees with it. Answers, from data, whether 409 is
    the only anomaly in the corpus."""
    print("\n[audit] ground-truth labelling (PHQ8_Score >= 10)")
    if gt is None:
        print("  [skip] GT CSVs not found."); return
    g = gt.copy()
    g["gt_diag_rule"] = g["PHQ8_Score"].apply(phq8_to_diagnosis)
    rows = []
    if "PHQ8_Binary" in g.columns:
        g["official"] = pd.to_numeric(g["PHQ8_Binary"], errors="coerce")
        dis = g[g["gt_diag_rule"] != g["official"]]
        if len(dis) == 0:
            print("  no subject disagrees with the official PHQ8_Binary column.")
        for r in dis.itertuples():
            print(f"  DISAGREES: subject {int(r.Participant_ID)}  score={int(r.PHQ8_Score)}  "
                  f"official={int(r.official)}  rule={int(r.gt_diag_rule)}")
            rows.append({"Participant_ID": int(r.Participant_ID),
                         "PHQ8_Score": int(r.PHQ8_Score),
                         "official_PHQ8_Binary": int(r.official),
                         "rule_score_ge_10": int(r.gt_diag_rule)})
    else:
        print("  [warn] no PHQ8_Binary column in GT -> cannot audit disagreements.")
    if rows:
        save_table(pd.DataFrame(rows), "A0_label_disagreements", outdir)

    # class balance per split, under the >=10 rule (for thesis Table 4.1)
    meta_csv = cfg.get("meta_zeroshot_csv")
    if _exists(meta_csv):
        test_ids = set(load_meta(meta_csv)["participant_id"].astype(int))
        t = g[g["Participant_ID"].isin(test_ids)]
        n_pos = int(t["gt_diag_rule"].sum())
        print(f"  test split: {n_pos} depressed / {len(t)-n_pos} not depressed "
              f"(n={len(t)})")
        save_table(pd.DataFrame([{"split": "test", "n": len(t),
                                  "depressed": n_pos,
                                  "not_depressed": len(t) - n_pos}]),
                   "A0_class_balance_test", outdir)

def coverage_and_distribution(cfg, outdir):
    print("\n[no-GT] coverage, distribution, precomputed summary")
    cov = {}
    for cond, src in [("zero_shot", cfg["quant_zeroshot_csv"]),
                      ("few_shot",  cfg["quant_fewshot_jsonl"])]:
        if not _exists(src): continue
        q = load_quant(src)
        cov[cond] = {it: float(q[it].isna().mean()) for it in PHQ8_ITEMS}
        cov[cond]["OVERALL_item_na_rate"] = float(q[PHQ8_ITEMS].isna().mean().mean())
    if cov:
        save_table(pd.DataFrame(cov).reset_index().rename(columns={"index": "item"}),
                   "A0_na_rate_by_item", outdir)

    dist = {}
    for cond, src in [("zero_shot", cfg["meta_zeroshot_csv"]),
                      ("few_shot",  cfg["meta_fewshot_csv"])]:
        if not _exists(src): continue
        dist[cond] = load_meta(src)["severity"].astype(int).value_counts().sort_index()
    if dist:
        save_table(pd.DataFrame(dist).fillna(0).astype(int)
                   .reset_index().rename(columns={"index": "severity"}),
                   "A0_severity_distribution", outdir)

    if _exists(cfg["quant_zeroshot_csv"]):
        z = pd.read_csv(cfg["quant_zeroshot_csv"])
        s = {}
        if "avg_difference" in z:      s["mean_avg_difference (per-subj MAE on answered)"] = z["avg_difference"].mean()
        if "overall_accuracy" in z:    s["mean_overall_accuracy"] = z["overall_accuracy"].mean()
        if "accuracy_on_available" in z: s["mean_accuracy_on_available"] = z["accuracy_on_available"].mean()
        if "num_questions_na" in z:    s["mean_num_questions_na (of 8)"] = z["num_questions_na"].mean()
        if s:
            save_table(pd.DataFrame([s]), "A0_zeroshot_precomputed_summary", outdir)

def part_A_vs_gt(cfg, outdir, gt):
    print("\n[Part A] vs ground truth")
    if gt is None:
        print("  [skip] GT CSVs not found -> set CONFIG['gt_csvs']. "
              "Binary metrics, confusion matrices and MAE need GT.")
        return {}

    has_item_gt = all(c in gt.columns for c in PHQ8_ITEMS)
    bin_rows, mae_rows, peritem, ci_rows = [], [], {}, []

    for cond, meta_csv, quant_src in [
            ("zero_shot", cfg["meta_zeroshot_csv"], cfg["quant_zeroshot_csv"]),
            ("few_shot",  cfg["meta_fewshot_csv"],  cfg["quant_fewshot_jsonl"])]:

        if _exists(meta_csv):
            # Ground-truth binary = PHQ-8 total >= 10 (clinical screening rule).
            # NOT the official PHQ8_Binary column: for subject 409 that column
            # score with the >= 10 rule, so this keeps the comparison fair.
            keep = ["Participant_ID", "PHQ8_Score"]
            meta = load_meta(meta_csv).merge(
                gt[keep], left_on="participant_id", right_on="Participant_ID", how="inner")
            meta["gt_sev"]  = meta["PHQ8_Score"].apply(phq8_to_severity)
            meta["gt_diag"] = meta["PHQ8_Score"].apply(phq8_to_diagnosis)
            meta["pred_diag"] = meta["severity"].astype(int).apply(severity_to_diagnosis)

            n_pos = int(meta["gt_diag"].sum())
            print(f"  [{cond}] ground truth: {n_pos} depressed / {len(meta)-n_pos} "
                  f"not depressed  (rule: PHQ8_Score >= 10)")

            bm = binary_metrics(meta["gt_diag"], meta["pred_diag"]); bm["condition"] = cond; bm["n"] = len(meta)
            bin_rows.append(bm)

            from sklearn.metrics import confusion_matrix as _cm2
            tn, fp, fn, tp = _cm2(meta["gt_diag"], meta["pred_diag"], labels=[0, 1]).ravel()
            for mname, k, nn in [("accuracy",    tp + tn, tp + tn + fp + fn),
                                 ("recall",      tp,      tp + fn),
                                 ("precision",   tp,      tp + fp),
                                 ("specificity", tn,      tn + fp)]:
                lo, hi = wilson_ci(int(k), int(nn))
                ci_rows.append({"condition": cond, "metric": mname,
                                "point": (k / nn if nn else np.nan),
                                "ci95_low": lo, "ci95_high": hi, "n_denominator": int(nn)})

            # per-subject error list -- the wrong diagnoses, for the thesis text
            errs = meta[meta["gt_diag"] != meta["pred_diag"]].copy()
            errs["error_type"] = np.where(errs["gt_diag"] == 1, "FN (missed)", "FP (over-called)")
            if len(errs):
                save_table(errs[["participant_id", "PHQ8_Score", "gt_diag",
                                 "severity", "pred_diag", "error_type"]]
                           .sort_values("participant_id"),
                           f"A_errors_{cond}", outdir)
                print(f"  [{cond}] wrong diagnoses ({len(errs)}): "
                      + ", ".join(f"{int(r.participant_id)}({r.error_type.split()[0]},"
                                  f"total={int(r.PHQ8_Score)})" for r in errs.itertuples()))

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(meta["gt_sev"], meta["severity"].astype(int), labels=[0,1,2,3,4])
            labs = ["Minimal","Mild","Moderate","Mod-Sev","Severe"]
            fig, ax = plt.subplots(figsize=(4.6,3.9))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labs, yticklabels=labs, ax=ax)
            ax.set_xlabel("Prediction"); ax.set_ylabel("Ground truth"); ax.set_title(f"Severity confusion ({cond})")
            save_fig(fig, f"A_confusion_severity_{cond}", outdir)

            # labels=[1,0] puts Depressed first on both axes, so the matrix reads
            # TP (top-left), FN, FP, TN, with the diagonal running TP -> TN.
            cm2 = confusion_matrix(meta["gt_diag"], meta["pred_diag"], labels=[1, 0])
            tp2, fn2, fp2, tn2 = cm2.ravel()
            blabs = ["Depressed", "Not depressed"]
            annot = np.array([[f"TP\n{tp2}", f"FN\n{fn2}"], [f"FP\n{fp2}", f"TN\n{tn2}"]])
            fig, ax = plt.subplots(figsize=(4.0, 3.4))
            sns.heatmap(cm2, annot=annot, fmt="", cmap="Blues", cbar=False,
                        xticklabels=blabs, yticklabels=blabs, ax=ax)
            ax.set_xlabel("Prediction"); ax.set_ylabel("Ground truth")
            ax.set_title(f"Binary confusion ({cond})")
            save_fig(fig, f"A_confusion_binary_{cond}", outdir)
            print(f"  [{cond}] binary: TP={tp2} FN={fn2} FP={fp2} TN={tn2}  "
                  f"acc={(tp2+tn2)/max(tp2+tn2+fp2+fn2,1):.4f}   (FP = over-diagnosis, FN = missed)")

        if _exists(quant_src) and has_item_gt:
            q = load_quant(quant_src).merge(gt[["Participant_ID"]+PHQ8_ITEMS],
                                            left_on="participant_id", right_on="Participant_ID",
                                            how="inner", suffixes=("", "_gt"))
            per, cover, all_err = {}, {}, []
            for it in PHQ8_ITEMS:
                err = (q[it] - q[it+"_gt"]).abs()
                per[it] = float(err.mean()) if err.notna().any() else np.nan
                cover[it] = int(err.notna().sum())
                all_err.append(err.dropna())
            peritem[cond] = per
            per_q_vals = [v for v in per.values() if not np.isnan(v)]
            mae_rows.append({"condition": cond,
                             "MAE_per_symptom_macro_PAPER": float(np.mean(per_q_vals)) if per_q_vals else np.nan,
                             "MAE_per_item_pooled": float(pd.concat(all_err).mean()) if all_err else np.nan,
                             "n_symptoms_scored": len(per_q_vals),
                             "n_item_predictions": int(pd.concat(all_err).shape[0]) if all_err else 0})
            # per-item coverage + MAE -> thesis appendix table
            save_table(pd.DataFrame([{"item": it, "scored": cover[it],
                                      "blank": len(q) - cover[it],
                                      "MAE": per[it]} for it in PHQ8_ITEMS]),
                       f"A_coverage_and_mae_{cond}", outdir)
        elif _exists(quant_src) and not has_item_gt:
            print(f"  [note] GT has no per-item columns -> per-item MAE skipped for {cond}.")

    out = {}
    if bin_rows:
        t = pd.DataFrame(bin_rows)[["condition","n","accuracy","balanced_accuracy","precision","recall","f1"]]
        save_table(t, "A_binary_metrics", outdir); out["binary"] = t
    if ci_rows:
        save_table(pd.DataFrame(ci_rows)[["condition","metric","point","ci95_low","ci95_high","n_denominator"]],
                   "A_binary_metrics_ci", outdir)
    if mae_rows:
        save_table(pd.DataFrame(mae_rows), "A_quant_mae", outdir); out["mae"] = pd.DataFrame(mae_rows)
    if peritem:
        pm = pd.DataFrame(peritem).reset_index().rename(columns={"index":"item"})
        save_table(pm, "A_quant_mae_by_item", outdir)
        fig, ax = plt.subplots(figsize=(7,3.6))
        pm2 = pm.set_index("item"); pm2.plot(kind="bar", ax=ax)
        ax.set_ylabel("MAE (answered items)"); ax.set_title("Per-item PHQ-8 MAE"); ax.legend(title="")
        plt.xticks(rotation=40, ha="right"); save_fig(fig, "A_quant_mae_by_item", outdir)
    return out

def part_B_vs_paper(cfg, outdir, A):
    print("\n[Part B] reported vs reproduced (paper comparison)")
    p = PAPER_REPORTED
    def _b(metric, cond):
        if "binary" in A and cond in A["binary"]["condition"].values:
            return A["binary"].set_index("condition").loc[cond, metric]
        return np.nan
    rows = [
        ("Accuracy",          p["binary_accuracy"],          p["human_accuracy"],
         _b("accuracy","zero_shot"),          _b("accuracy","few_shot")),
        ("Balanced accuracy", p["binary_balanced_accuracy"], p["human_balanced_accuracy"],
         _b("balanced_accuracy","zero_shot"), _b("balanced_accuracy","few_shot")),
        ("Precision",         p["binary_precision"],         p["human_precision"],
         _b("precision","zero_shot"),         _b("precision","few_shot")),
        ("Recall",            p["binary_recall"],            p["human_recall"],
         _b("recall","zero_shot"),            _b("recall","few_shot")),
        ("F1",                p["binary_f1"],                p["human_f1"],
         _b("f1","zero_shot"),                _b("f1","few_shot")),
    ]
    tbl = pd.DataFrame(rows, columns=["metric", "paper_agent", "paper_human",
                                      "reproduced_zero_shot", "reproduced_few_shot"])
    save_table(tbl, "B_reported_vs_reproduced", outdir)
    print("\n  ---- thesis Table 8.1 (2 dp) ----")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    def _mae(cond):
        if "mae" in A and cond in A["mae"]["condition"].values:
            return float(A["mae"].set_index("condition").loc[cond, "MAE_per_symptom_macro_PAPER"])
        return np.nan
    mae_cmp = pd.DataFrame([
        {"condition": "zero_shot", "paper_reported": p["quant_mae_zeroshot"], "reproduced_per_symptom_macro": _mae("zero_shot")},
        {"condition": "few_shot",  "paper_reported": p["quant_mae_fewshot"],  "reproduced_per_symptom_macro": _mae("few_shot")},
    ])
    save_table(mae_cmp, "B_mae_reported_vs_reproduced", outdir)
    save_table(pd.DataFrame([
        {"metric": "No-prediction rate (paper)",       "paper_reported": p["no_prediction_rate"]},
        {"metric": "Human-agent agreement (paper)",    "paper_reported": p["human_agent_agreement"]},
    ]), "B_paper_reference_values", outdir)

    if "binary" in A:
        plot_df = tbl.set_index("metric")[["paper_agent", "paper_human",
                                           "reproduced_zero_shot", "reproduced_few_shot"]]
        fig, ax = plt.subplots(figsize=(7.6,3.8)); plot_df.plot(kind="bar", ax=ax)
        ax.set_ylim(0,1.05); ax.set_ylabel("score"); ax.set_title("Reported vs reproduced (binary)")
        ax.legend(title="", fontsize=8); plt.xticks(rotation=20, ha="right")
        save_fig(fig, "B_reported_vs_reproduced", outdir)

def main():
    outdir = CONFIG["output_dir"]; os.makedirs(outdir, exist_ok=True)
    print(f"Writing to: {os.path.abspath(outdir)}")
    coverage_and_distribution(CONFIG, outdir)
    gt = load_gt(CONFIG)
    label_audit(gt, CONFIG, outdir)
    A = part_A_vs_gt(CONFIG, outdir, gt)
    part_B_vs_paper(CONFIG, outdir, A)
    print("\nDone.")

if __name__ == "__main__":
    main()
