

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.join(SCRIPT_DIR, "..")
SYMP_CSV   = os.path.join(ROOT, "analysis_output", "rq3", "rq3_main_symptoms.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "output")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
TAB_DIR    = os.path.join(OUT_DIR, "tables")

for d in [FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

PHQ8_LABELS = {
    "PHQ8_NoInterest":    "Anhedonia",
    "PHQ8_Depressed":     "Depressed mood",
    "PHQ8_Sleep":         "Sleep issues",
    "PHQ8_Tired":         "Fatigue",
    "PHQ8_Appetite":      "Appetite issues",
    "PHQ8_Failure":       "Feelings of failure",
    "PHQ8_Concentrating": "Concentration problems",
    "PHQ8_Moving":        "Psychomotor changes"
}

PHQ8_LABELS_WRAP = {
    "PHQ8_NoInterest":    "No Interest\n(Anhedonia)",
    "PHQ8_Depressed":     "Depressed\nMood",
    "PHQ8_Sleep":         "Sleep\nProblems",
    "PHQ8_Tired":         "Fatigue",
    "PHQ8_Appetite":      "Appetite\nChanges",
    "PHQ8_Failure":       "Feelings of\nFailure",
    "PHQ8_Concentrating": "Concentration\nProblems",
    "PHQ8_Moving":        "Psychomotor\nChanges",
}

def load_symptoms():
    df = pd.read_csv(SYMP_CSV, engine="python", on_bad_lines="skip")
    df = df[pd.to_numeric(df["pid"], errors="coerce").notna()].copy()
    df["pid"] = df["pid"].astype(int)
    
    
    bool_map = {'True': True, 'False': False, 'true': True, 'false': False, True: True, False: False}
    df["exact_correct"]  = df["exact_correct"].map(bool_map)
    df["within1_correct"]= df["within1_correct"].map(bool_map)
    
    for col in ["token_prob", "token_p_norm", "pred_score", "gt_score", "s2_symptom_conf"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def compute_symptom_stats(df):
    rows = []
    for sym, sym_label in PHQ8_LABELS.items():
        sub = df[df["symptom"] == sym].copy()
        sub_gt = sub[sub["gt_score"].notna()]
        n_total  = len(sub)
        n_with_gt= len(sub_gt)

        exact_acc  = sub_gt["exact_correct"].mean()  if n_with_gt > 0 else np.nan
        within1_acc= sub_gt["within1_correct"].mean() if n_with_gt > 0 else np.nan

        mean_tp = sub["token_prob"].mean()
        std_tp  = sub["token_prob"].std()

        if n_with_gt > 2 and sub_gt["token_prob"].std() > 0:
            rho, pval = spearmanr(
                sub_gt["token_prob"].fillna(sub_gt["token_prob"].mean()),
                sub_gt["exact_correct"].astype(int)
            )
        else:
            rho, pval = np.nan, np.nan

        
        tp_correct = sub_gt.loc[sub_gt["exact_correct"] == True,  "token_prob"].mean()
        tp_wrong   = sub_gt.loc[sub_gt["exact_correct"] == False, "token_prob"].mean()

        rows.append({
            "symptom":       sym,
            "label":         sym_label,
            "n_scored":      n_total,
            "n_with_gt":     n_with_gt,
            "exact_acc":     round(exact_acc,   4) if not np.isnan(exact_acc)   else np.nan,
            "within1_acc":   round(within1_acc, 4) if not np.isnan(within1_acc) else np.nan,
            "mean_token_prob":  round(mean_tp,  4) if not np.isnan(mean_tp)  else np.nan,
            "std_token_prob":   round(std_tp,   4) if not np.isnan(std_tp)   else np.nan,
            "spearman_rho":     round(rho,   4) if not np.isnan(rho)   else np.nan,
            "spearman_pval":    round(pval,  4) if not np.isnan(pval)  else np.nan,
            "mean_tp_correct":  round(tp_correct, 4) if not np.isnan(tp_correct) else np.nan,
            "mean_tp_wrong":    round(tp_wrong,   4) if not np.isnan(tp_wrong)   else np.nan,
        })
    return pd.DataFrame(rows)

def compute_pooled_symptom_calibration(df):
    """Computes Table 7.15 (pooled symptom calibration metrics)."""
    from sklearn.metrics import roc_auc_score, brier_score_loss

    df = df.copy()
    df["s2_norm"] = df["s2_symptom_conf"] / 100.0

    def ece(y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            if i == n_bins - 1:
                mask = (y_prob >= lo) & (y_prob <= hi)
            else:
                mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            acc  = y_true[mask].mean()
            conf = y_prob[mask].mean()
            frac = mask.sum() / len(y_true)
            ece_val += frac * abs(acc - conf)
        return ece_val

    
    sub_s1_ex = df.dropna(subset=["token_prob", "exact_correct"])
    y_true_s1_ex = sub_s1_ex["exact_correct"].values.astype(int)
    y_prob_s1_ex = sub_s1_ex["token_prob"].values
    s1_ex_brier = brier_score_loss(y_true_s1_ex, y_prob_s1_ex)
    s1_ex_auc   = roc_auc_score(y_true_s1_ex, y_prob_s1_ex)
    s1_ex_ece   = ece(y_true_s1_ex, y_prob_s1_ex, 10)

    
    sub_s1_w1 = df.dropna(subset=["token_prob", "within1_correct"])
    y_true_s1_w1 = sub_s1_w1["within1_correct"].values.astype(int)
    y_prob_s1_w1 = sub_s1_w1["token_prob"].values
    s1_w1_brier = brier_score_loss(y_true_s1_w1, y_prob_s1_w1)
    s1_w1_auc   = roc_auc_score(y_true_s1_w1, y_prob_s1_w1)
    s1_w1_ece   = ece(y_true_s1_w1, y_prob_s1_w1, 10)

   
    sub_s2_ex = df.dropna(subset=["s2_norm", "exact_correct"])
    y_true_s2_ex = sub_s2_ex["exact_correct"].values.astype(int)
    y_prob_s2_ex = sub_s2_ex["s2_norm"].values
    s2_ex_brier = brier_score_loss(y_true_s2_ex, y_prob_s2_ex)
    s2_ex_auc   = roc_auc_score(y_true_s2_ex, y_prob_s2_ex)
    s2_ex_ece   = ece(y_true_s2_ex, y_prob_s2_ex, 10)

    return pd.DataFrame([
        {
            "Signal": "S1 Token Prob vs Exact Correct",
            "N": len(y_true_s1_ex),
            "Brier Score": round(s1_ex_brier, 4),
            "AUROC": round(s1_ex_auc, 4),
            "ECE": round(s1_ex_ece, 4)
        },
        {
            "Signal": "S1 Token Prob vs Within-1 Correct",
            "N": len(y_true_s1_w1),
            "Brier Score": round(s1_w1_brier, 4),
            "AUROC": round(s1_w1_auc, 4),
            "ECE": round(s1_w1_ece, 4)
        },
        {
            "Signal": "S2 Verbalized vs Exact Correct",
            "N": len(y_true_s2_ex),
            "Brier Score": round(s2_ex_brier, 4),
            "AUROC": round(s2_ex_auc, 4),
            "ECE": round(s2_ex_ece, 4)
        }
    ])

def plot_accuracy_bars(stats_df, filename):
    syms    = stats_df["symptom"].tolist()
    labels  = [PHQ8_LABELS_WRAP[s] for s in syms]
    exact   = stats_df["exact_acc"].fillna(0).tolist()
    within1 = stats_df["within1_acc"].fillna(0).tolist()
    x = np.arange(len(syms))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, exact,   w, label="Exact accuracy",    color="#3498db", alpha=0.85)
    ax.bar(x + w/2, within1, w, label="Within-1 accuracy", color="#2ecc71", alpha=0.85)

    for i, (e, w1) in enumerate(zip(exact, within1)):
        ax.text(i - w/2, e  + 0.01, f"{e:.0%}",  ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w/2, w1 + 0.01, f"{w1:.0%}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Per-Symptom PHQ-8 Score Accuracy (Main Condition)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [saved] {os.path.basename(filename)}")

def plot_logprob_vs_accuracy(stats_df, filename):
    valid = stats_df[stats_df["exact_acc"].notna() & stats_df["mean_token_prob"].notna()]
    x   = valid["mean_token_prob"].values
    y   = valid["exact_acc"].values
    n   = valid["n_with_gt"].values
    syms= [PHQ8_LABELS[s] for s in valid["symptom"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x, y, s=n * 8, c=y, cmap="RdYlGn", vmin=0, vmax=1,
                    alpha=0.85, edgecolors="grey", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Exact Accuracy")

    for xi, yi, sym, ni in zip(x, y, syms, n):
        ax.annotate(sym, (xi, yi), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5, color="#2c3e50")

    ax.set_xlabel("Mean Token Log-Probability (S1)", fontsize=11)
    ax.set_ylabel("Exact Accuracy", fontsize=11)
    ax.set_title("Per-Symptom: Token Log-Probability vs. Exact Accuracy\n"
                 "(bubble size = number of subjects with ground truth)", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [saved] {os.path.basename(filename)}")

def plot_token_prob_by_correctness(df, filename):
    syms   = list(PHQ8_LABELS.keys())
    n_sym  = len(syms)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    axes_flat = axes.flatten()

    for ax, sym in zip(axes_flat, syms):
        sub = df[df["symptom"] == sym].dropna(subset=["token_prob", "exact_correct"])
        if sub.empty:
            ax.set_title(PHQ8_LABELS[sym], fontsize=8)
            ax.set_visible(False)
            continue

        correct_tp = sub.loc[sub["exact_correct"] == True,  "token_prob"].values
        wrong_tp   = sub.loc[sub["exact_correct"] == False, "token_prob"].values

        parts = ax.violinplot(
            [correct_tp if len(correct_tp) > 1 else [np.nan],
             wrong_tp   if len(wrong_tp)   > 1 else [np.nan]],
            positions=[0, 1], showmedians=True, widths=0.6
        )
        for pc, color in zip(parts["bodies"], ["#2ecc71", "#e74c3c"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)

        if len(correct_tp) > 0:
            ax.scatter(np.random.normal(0, 0.04, len(correct_tp)), correct_tp,
                       color="#27ae60", alpha=0.6, s=14, zorder=3)
        if len(wrong_tp) > 0:
            ax.scatter(np.random.normal(1, 0.04, len(wrong_tp)), wrong_tp,
                       color="#c0392b", alpha=0.6, s=14, zorder=3)

        ax.set_title(PHQ8_LABELS[sym], fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Correct", "Wrong"], fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)

    axes_flat[0].set_ylabel("Token Log-Probability (S1)", fontsize=10)
    axes_flat[4].set_ylabel("Token Log-Probability (S1)", fontsize=10)

    fig.suptitle("Token Log-Probability Distribution by Correctness per PHQ-8 Symptom\n"
                 "(Main Condition — If well-calibrated, correct preds should have higher S1)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {os.path.basename(filename)}")

def main():
    print("=" * 64)
    print("RQ3 SYMPTOM-LEVEL CALIBRATION ANALYSIS (v2)")
    print("=" * 64)

    df = load_symptoms()
    print(f"Loaded {len(df)} symptom rows, covering {df['pid'].nunique()} subjects.\n")

    stats_df = compute_symptom_stats(df)
    pooled_df = compute_pooled_symptom_calibration(df)

    print("── Per-Symptom Stats ────────────────────────────────────────")
    for _, row in stats_df.iterrows():
        rho_str = (f"Spearman ρ={row['spearman_rho']:.3f} (p={row['spearman_pval']:.3f})"
                   if not np.isnan(row["spearman_rho"]) else "Spearman: n/a")
        overconf = ""
        if not (np.isnan(row["mean_tp_correct"]) or np.isnan(row["mean_tp_wrong"])):
            diff = row["mean_tp_correct"] - row["mean_tp_wrong"]
            overconf = (f" | mean_tp: correct={row['mean_tp_correct']:.3f}, "
                        f"wrong={row['mean_tp_wrong']:.3f} (Δ={diff:+.3f})")
        print(f"  {row['symptom']:25s}  exact={row['exact_acc']:.1%}  "
              f"w1={row['within1_acc']:.1%}  mean_tp={row['mean_token_prob']:.3f}  "
              f"{rho_str}{overconf}")

    print("\n── Key Calibration Observation ──────────────────────────────")
    for _, row in stats_df.iterrows():
        if not (np.isnan(row["mean_tp_correct"]) or np.isnan(row["mean_tp_wrong"])):
            diff = row["mean_tp_correct"] - row["mean_tp_wrong"]
            if diff < 0:
                print(f"  [MISCALIBRATED] {row['symptom']}: token_prob is HIGHER for WRONG "
                      f"predictions (Δ={diff:+.3f}) — model more confident when wrong!")
            elif diff > 0.05:
                print(f"  [PARTLY CALIBRATED] {row['symptom']}: token_prob higher for correct "
                      f"(Δ={diff:+.3f})")
            else:
                print(f"  [FLAT] {row['symptom']}: almost no difference in token_prob "
                      f"(Δ={diff:+.3f})")

    print("\n── Pooled Symptom Calibration Table (Table 7.15) ────────────")
    print(pooled_df.to_string(index=False))

    print("\n── Generating plots ─────────────────────────────────────────")
    plot_accuracy_bars(
        stats_df,
        os.path.join(FIG_DIR, "symptom_accuracy_bars.png")
    )
    plot_logprob_vs_accuracy(
        stats_df,
        os.path.join(FIG_DIR, "symptom_logprob_vs_accuracy.png")
    )
    plot_token_prob_by_correctness(
        df,
        os.path.join(FIG_DIR, "symptom_token_prob_distribution.png")
    )

    out1 = os.path.join(TAB_DIR, "symptom_accuracy_table.csv")
    out2 = os.path.join(TAB_DIR, "symptom_logprob_calibration.csv")
    out3 = os.path.join(TAB_DIR, "pooled_symptom_calibration.csv")
    
    stats_df[["symptom", "label", "n_scored", "n_with_gt", "exact_acc", "within1_acc"]
             ].to_csv(out1, index=False)
    stats_df[["symptom", "label", "mean_token_prob", "std_token_prob",
              "spearman_rho", "spearman_pval", "mean_tp_correct", "mean_tp_wrong"]
             ].to_csv(out2, index=False)
    pooled_df.to_csv(out3, index=False)
    
    print(f"\n[saved] symptom_accuracy_table.csv")
    print(f"[saved] symptom_logprob_calibration.csv")
    print(f"[saved] pooled_symptom_calibration.csv")

    print("\nDone. Figures in:", FIG_DIR)
    print("Tables in:", TAB_DIR)

if __name__ == "__main__":
    main()
