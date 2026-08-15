

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rq3_labels import apply_gt_fix

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.join(SCRIPT_DIR, "..")
DATA_DIR   = os.path.join(ROOT, "analysis_output", "rq3")
OUT_DIR    = os.path.join(SCRIPT_DIR, "output")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
TAB_DIR    = os.path.join(OUT_DIR, "tables")

for d in [FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

CONDITIONS = [
    ("main",              "Main\n(full pipeline)"),
    ("no_qual",           "No Qual\n(−qualitative)"),
    ("no_quant",          "No Quant\n(−quantitative)"),
    ("transcript_only",   "Transcript\nOnly"),
    ("explanation_first", "Explanation\nFirst"),
]


COND_COLOURS = {
    "main":              "#2ecc71",
    "no_qual":           "#e67e22",
    "no_quant":          "#3498db",
    "transcript_only":   "#e74c3c",
    "explanation_first": "#9b59b6",
}

def load_condition(cond_key):
    path = os.path.join(DATA_DIR, f"rq3_{cond_key}_meta.csv")
    if not os.path.exists(path):
        print(f"  [WARN] file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df[pd.to_numeric(df["pid"], errors="coerce").notna()].copy()
    df["pid"] = df["pid"].astype(int)
    
    for col in ["s1_p_depressed", "s3_agree_binary", "s4_mean_p_depressed", "binary", "gt_binary"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = apply_gt_fix(df)
        
    df["correct_bool"] = df["binary"] == df["gt_binary"]
    
    df["s1_conf"] = np.where(df["binary"] == 1, df["s1_p_depressed"], 1.0 - df["s1_p_depressed"])
    df["s4_conf"] = np.where(df["binary"] == 1, df["s4_mean_p_depressed"], 1.0 - df["s4_mean_p_depressed"])
    
    return df

def summarise_condition(cond_key, label, df):
    if df.empty:
        return None
    n = len(df)
    acc         = df["correct_bool"].mean()
    s1_mean     = df["s1_conf"].mean()
    s1_std      = df["s1_conf"].std()
    s3_mean     = df["s3_agree_binary"].mean()
    s3_std      = df["s3_agree_binary"].std()
    s4_mean     = df["s4_conf"].mean()
    s4_std      = df["s4_conf"].std()
    
    s3_collapse = (df["s3_agree_binary"] == 1.0).mean()

    
    from sklearn.metrics import roc_auc_score
    try:
        s1_auc = roc_auc_score(df["correct_bool"].values.astype(int), df["s1_conf"].values)
    except Exception:
        s1_auc = np.nan

    return {
        "condition":    cond_key,
        "label":        label.replace("\n", " "),
        "n":            n,
        "accuracy":     round(acc,     4),
        "s1_mean":      round(s1_mean, 4),
        "s1_std":       round(s1_std,  4),
        "s1_auroc":     round(s1_auc,  4) if not np.isnan(s1_auc) else np.nan,
        "s3_mean":      round(s3_mean, 4),
        "s3_std":       round(s3_std,  4),
        "s3_collapse":  round(s3_collapse, 4),
        "s4_mean":      round(s4_mean, 4),
        "s4_std":       round(s4_std,  4),
    }

def bar_plot(cond_keys, labels, values, errors, ylabel, title, filename,
             refline=None, refline_label=None):
    x      = np.arange(len(cond_keys))
    colours= [COND_COLOURS[k] for k in cond_keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, values, yerr=errors, capsize=4, color=colours,
                  alpha=0.85, edgecolor="white", linewidth=0.7)

    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    if refline is not None:
        ax.axhline(refline, linestyle="--", color="#7f8c8d", linewidth=1.2,
                   label=refline_label or f"Reference = {refline}")
        ax.legend(fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, min(max(v for v in values if not np.isnan(v)) * 1.25, 1.05))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [saved] {os.path.basename(filename)}")

def plot_s3_collapse(summaries, filename):
    cond_keys  = [s["condition"] for s in summaries]
    labels     = [s["label"].replace("  ", "\n") for s in summaries]
    collapsed  = [s["s3_collapse"]      for s in summaries]
    varying    = [1.0 - s["s3_collapse"] for s in summaries]

    x      = np.arange(len(cond_keys))
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x, collapsed, color="#e74c3c", alpha=0.85, label="S3 = 1.0 (no variance)", edgecolor="white")
    ax.bar(x, varying, bottom=collapsed, color="#2ecc71", alpha=0.85,
           label="S3 < 1.0 (some variance)", edgecolor="white")

    for i, (c, v) in enumerate(zip(collapsed, varying)):
        ax.text(i, c / 2, f"{c:.0%}", ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold")
        if v > 0.02:
            ax.text(i, c + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Fraction of Subjects", fontsize=11)
    ax.set_title("S3 Sampling Agreement Collapse Rate per Condition\n"
                 "(S3=1.0 means all K samples agreed — signal is degenerate)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [saved] {os.path.basename(filename)}")

def plot_accuracy_with_chance(summaries, filename):
    cond_keys  = [s["condition"]  for s in summaries]
    labels     = [s["label"]      for s in summaries]
    accs       = [s["accuracy"]   for s in summaries]
    colours    = [COND_COLOURS[k] for k in cond_keys]

    x   = np.arange(len(cond_keys))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, accs, color=colours, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.5, linestyle=":", color="#7f8c8d", linewidth=1.2, label="50% reference")
    ax.axhline(accs[0], linestyle="--", color=COND_COLOURS["main"],
               linewidth=1.2, alpha=0.7, label="Main condition accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Binary Accuracy", fontsize=11)
    ax.set_title("Binary Classification Accuracy per Input Condition", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [saved] {os.path.basename(filename)}")

def main():
    print("=" * 64)
    print("RQ3 ABLATION COMPARISON — All 5 Conditions (v2)")
    print("=" * 64)

    all_data   = {}
    summaries  = []

    for cond_key, label in CONDITIONS:
        df = load_condition(cond_key)
        all_data[cond_key] = df
        s  = summarise_condition(cond_key, label, df)
        if s:
            summaries.append(s)
            print(f"\n── Condition: {cond_key} (n={s['n']})")
            print(f"   Accuracy                  : {s['accuracy']:.1%}")
            print(f"   S1 conf mean±std (Correctness): {s['s1_mean']:.4f} ± {s['s1_std']:.4f}")
            print(f"   S1 AUROC vs Correctness   : {s['s1_auroc']:.4f}" if not np.isnan(s['s1_auroc']) else "   S1 AUROC vs Correctness   : N/A")
            print(f"   S3 agreement mean±std     : {s['s3_mean']:.4f} ± {s['s3_std']:.4f}")
            print(f"   S3 collapse rate          : {s['s3_collapse']:.1%}")
            print(f"   S4 conf mean±std (Correctness): {s['s4_mean']:.4f} ± {s['s4_std']:.4f}")

    if not summaries:
        print("[ERROR] No condition data loaded. Check DATA_DIR path.")
        return

    cond_keys = [s["condition"]  for s in summaries]
    labels    = [s["label"]      for s in summaries]

    print("\n── Generating plots ─────────────────────────────────────────")

    plot_accuracy_with_chance(
        summaries,
        os.path.join(FIG_DIR, "ablation_accuracy.png")
    )

    bar_plot(
        cond_keys, labels,
        values=[s["s1_mean"] for s in summaries],
        errors=[s["s1_std"]  for s in summaries],
        ylabel="Mean S1 Correctness Confidence",
        title="S1 Correctness Token Logprob Confidence per Condition\n(high = model is certain of its own prediction)",
        filename=os.path.join(FIG_DIR, "ablation_s1_confidence.png"),
        refline=0.5, refline_label="Random chance confidence (0.5)"
    )

    bar_plot(
        cond_keys, labels,
        values=[s["s3_mean"] for s in summaries],
        errors=[s["s3_std"]  for s in summaries],
        ylabel="Mean S3 Sampling Agreement",
        title="S3 Stochastic Sampling Agreement per Condition\n"
              "(S3=1.0 for all subjects = degenerate, no discriminative value)",
        filename=os.path.join(FIG_DIR, "ablation_s3_agreement.png"),
        refline=1.0, refline_label="S3=1.0 (degenerate)"
    )

    plot_s3_collapse(
        summaries,
        os.path.join(FIG_DIR, "ablation_s3_collapse.png")
    )

    bar_plot(
        cond_keys, labels,
        values=[s["s4_mean"] for s in summaries],
        errors=[s["s4_std"]  for s in summaries],
        ylabel="Mean S4 Correctness Confidence",
        title="S4 Mean Correctness P(depressed) Confidence from Samples per Condition",
        filename=os.path.join(FIG_DIR, "ablation_s4_pdep.png"),
        refline=0.5, refline_label="Random chance confidence (0.5)"
    )

    summary_df = pd.DataFrame(summaries).drop(columns=["label"])
    out_csv    = os.path.join(TAB_DIR, "ablation_comparison.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[saved] ablation_comparison.csv")

    print("\n── Summary Table ────────────────────────────────────────────")
    print(summary_df.to_string(index=False))

    print("\n── Key observations ─────────────────────────────────────────")
    main_acc = next(s["accuracy"] for s in summaries if s["condition"] == "main")
    for s in summaries:
        delta = s["accuracy"] - main_acc
        tag   = "↓" if delta < -0.02 else ("↑" if delta > 0.02 else "≈")
        print(f"  {s['condition']:20s}  acc={s['accuracy']:.1%}  {tag} vs main ({delta:+.1%})"
              f"  S3_collapse={s['s3_collapse']:.0%}  S1_conf={s['s1_mean']:.3f}")

    print("\nDone. Figures in:", FIG_DIR)
    print("Tables in:", TAB_DIR)

if __name__ == "__main__":
    main()
