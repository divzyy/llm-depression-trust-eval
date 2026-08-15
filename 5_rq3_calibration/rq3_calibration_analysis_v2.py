

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

from rq3_labels import apply_gt_fix

warnings.filterwarnings("ignore")


plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.titlesize": 15,
})

FIG_DPI = 200

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.join(SCRIPT_DIR, "..")
DATA_DIR    = os.path.join(ROOT, "analysis_output", "rq3")
OUT_DIR     = os.path.join(SCRIPT_DIR, "output")
FIG_DIR     = os.path.join(OUT_DIR, "figures")
TAB_DIR     = os.path.join(OUT_DIR, "tables")

for d in [FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

MAIN_CSV    = os.path.join(DATA_DIR, "rq3_main_meta.csv")

C_CORRECT   = "#2ecc71"   # emerald green
C_WRONG     = "#e74c3c"   # alizarin red
C_CALIB     = "#9b59b6"   # amethyst purple (reliability line)
C_REF       = "#7f8c8d"   # asbestos grey  (perfect calibration diagonal)
C_ABOVE     = "#1abc9c"   # signal whose CI clears chance
C_CHANCE    = "#bdc3c7"   # signal whose CI includes chance

SIGNAL_META = {
    "S1 Token Logprob (Correctness)": ("s1_conf", "S1: Correctness Token Log-Probability"),
    "S2 Verbalized Conf.":            ("s2_conf_norm", "S2: Verbalized SToPS Confidence"),
    "S3 Sampling Agreement":          ("s3_agree_binary", "S3: Stochastic Sampling Agreement"),
    "S4 Mean P(depressed) (Correctness)": ("s4_conf", "S4: Correctness Mean P(depressed)"),
}

AUROC_LABELS = {
    "S1 Token Logprob (Correctness)":     "S1\ntoken probability",
    "S2 Verbalized Conf.":                "S2\nverbalised confidence",
    "S3 Sampling Agreement":              "S3\nsampling agreement",
    "S4 Mean P(depressed) (Correctness)": "S4\nmean predicted\nprobability",
}

def load_main_meta():
    """Load and process the main condition CSV to compute correctness-calibrated signals."""
    df = pd.read_csv(MAIN_CSV, engine="python", on_bad_lines="skip")
    df = df[pd.to_numeric(df["pid"], errors="coerce").notna()].copy()
    df["pid"] = df["pid"].astype(int)

    df["binary"] = pd.to_numeric(df["binary"], errors="coerce")
    df["gt_binary"] = pd.to_numeric(df["gt_binary"], errors="coerce")

    # Recompute the ground-truth label from the PHQ-8 rule (total >= 10)
    df = apply_gt_fix(df)

    df["correct_bool"] = df["binary"] == df["gt_binary"]
    df["y"] = df["correct_bool"].astype(int)

    # Normalize S2 from 0-100 to 0-1
    df["s2_conf_norm"] = pd.to_numeric(df["s2_conf"], errors="coerce") / 100.0

    for col in ["s1_p_depressed", "s3_agree_binary", "s4_mean_p_depressed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # S1 correctness-calibrated confidence:
    # If predicted depressed (binary=1), confidence is s1_p_depressed.
    # If predicted not depressed (binary=0), confidence is 1 - s1_p_depressed.
    df["s1_conf"] = np.where(df["binary"] == 1, df["s1_p_depressed"], 1.0 - df["s1_p_depressed"])

    # S4 correctness-calibrated confidence:
    # If predicted depressed (binary=1), confidence is s4_mean_p_depressed.
    # If predicted not depressed (binary=0), confidence is 1 - s4_mean_p_depressed.
    df["s4_conf"] = np.where(df["binary"] == 1, df["s4_mean_p_depressed"], 1.0 - df["s4_mean_p_depressed"])

    return df

def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error with equal-width bins, handling 1.0 boundary correctly."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    bin_data = []

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Last bin should be inclusive of the right endpoint (hi = 1.0)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        if mask.sum() == 0:
            bin_data.append((lo, hi, 0, np.nan, np.nan))
            continue

        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        frac = mask.sum() / len(y_true)
        ece_val += frac * abs(acc - conf)
        bin_data.append((lo, hi, mask.sum(), acc, conf))

    return ece_val, bin_data

def bootstrap_metric(y_true, y_prob, metric_fn, n_bootstrap=1000, rng_seed=42):
    """
    Compute point estimate and bootstrap 95% confidence intervals for a metric.
    Returns: point_estimate, ci_low, ci_high
    """
    valid = ~np.isnan(y_prob)
    yt, yp = y_true[valid], y_prob[valid]

    if len(yt) < 5:
        return np.nan, np.nan, np.nan

    try:
        pe = metric_fn(yt, yp)
    except Exception:
        pe = np.nan

    rng = np.random.default_rng(rng_seed)
    n_samples = len(yt)
    boot_stats = []

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        yt_boot = yt[boot_idx]
        yp_boot = yp[boot_idx]

        try:
            val = metric_fn(yt_boot, yp_boot)
            if not np.isnan(val):
                boot_stats.append(val)
        except Exception:
            continue

    if len(boot_stats) == 0:
        return pe, np.nan, np.nan

    ci_low = np.percentile(boot_stats, 2.5)
    ci_high = np.percentile(boot_stats, 97.5)
    return pe, ci_low, ci_high

def compute_all_metrics_with_ci(y_true, y_prob, label):
    """Compute point estimates and 95% bootstrap CIs for Brier, AUROC, ECE."""
    brier_pe, brier_l, brier_h = bootstrap_metric(y_true, y_prob, lambda yt, yp: brier_score_loss(yt, yp))
    auroc_pe, auroc_l, auroc_h = bootstrap_metric(y_true, y_prob, lambda yt, yp: roc_auc_score(yt, yp))
    ece_pe, ece_l, ece_h = bootstrap_metric(y_true, y_prob, lambda yt, yp: ece(yt, yp)[0])

    valid = ~np.isnan(y_prob)
    yt, yp = y_true[valid], y_prob[valid]
    c_correct = yp[yt == 1].mean() if (yt == 1).sum() > 0 else np.nan
    c_wrong   = yp[yt == 0].mean() if (yt == 0).sum() > 0 else np.nan

    return {
        "signal": label,
        "n": len(yt),
        "brier": brier_pe, "brier_ci_low": brier_l, "brier_ci_high": brier_h,
        "auroc": auroc_pe, "auroc_ci_low": auroc_l, "auroc_ci_high": auroc_h,
        "ece": ece_pe, "ece_ci_low": ece_l, "ece_ci_high": ece_h,
        "mean_conf_correct": c_correct,
        "mean_conf_wrong": c_wrong
    }

def plot_reliability_diagram(y_true, y_prob, signal_label, filename, n_bins=10):
    """Reliability diagram with binned accuracies and counts."""
    valid = ~np.isnan(y_prob)
    yt, yp = y_true[valid], y_prob[valid]
    ece_val, bin_data = ece(yt, yp, n_bins)

    bin_centres, accs, confs, counts = [], [], [], []
    for lo, hi, cnt, acc, conf in bin_data:
        if cnt > 0 and not np.isnan(acc):
            bin_centres.append((lo + hi) / 2)
            accs.append(acc)
            confs.append(conf)
            counts.append(cnt)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([0, 1], [0, 1], "--", color=C_REF, linewidth=1.5, label="Perfect calibration")

    if bin_centres:
        ax.plot(confs, accs, "o-", color=C_CALIB, markersize=9, linewidth=2.2, label="Calibration curve")
        for c, x_pos, y_pos in zip(counts, confs, accs):
            ax.annotate(f"n={c}", (x_pos, y_pos), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=11, color="#34495e",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="#bdc3c7"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence (correctness-oriented)", fontsize=14)
    ax.set_ylabel("Observed accuracy (fraction correct)", fontsize=14)
    ax.set_title(f"ECE = {ece_val:.4f}", fontsize=14)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [saved] {os.path.basename(filename)}")

def plot_confidence_distribution(y_true, y_prob, signal_label, filename):
    """Grouped bar chart of confidence, split by correct and incorrect.

    Bars sit at their true position on the 0-1 confidence axis, and all ten
    bins are drawn even when empty. The empty bins are part of the finding:
    they show that the signal never takes a middle value. No title is drawn,
    because the thesis caption carries that text.
    """
    valid = ~np.isnan(y_prob)
    yt, yp = y_true[valid], y_prob[valid]

    correct_conf = yp[yt == 1]
    wrong_conf   = yp[yt == 0]

    n_bins  = 10
    edges   = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    correct_counts = np.zeros(n_bins, dtype=int)
    wrong_counts   = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:          # last bin includes the 1.0 endpoint
            c_mask = (correct_conf >= lo) & (correct_conf <= hi)
            w_mask = (wrong_conf   >= lo) & (wrong_conf   <= hi)
        else:
            c_mask = (correct_conf >= lo) & (correct_conf < hi)
            w_mask = (wrong_conf   >= lo) & (wrong_conf   < hi)
        correct_counts[i] = int(c_mask.sum())
        wrong_counts[i]   = int(w_mask.sum())

    top = max(correct_counts.max(), wrong_counts.max())
    if top == 0:
        plt.close("all")
        return

    w = 0.042   # two bars side by side inside a 0.1-wide bin

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_c = ax.bar(centres - w / 2, correct_counts, w, color=C_CORRECT,
                    edgecolor="white", linewidth=0.8,
                    label=f"Correct (n={len(correct_conf)})")
    bars_w = ax.bar(centres + w / 2, wrong_counts, w, color=C_WRONG,
                    edgecolor="white", linewidth=0.8,
                    label=f"Incorrect (n={len(wrong_conf)})")

    for bars in (bars_c, bars_w):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + top * 0.015,
                        str(int(h)), ha="center", va="bottom",
                        fontsize=12, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e:.1f}" for e in edges], fontsize=12)
    ax.set_xlabel("Confidence score", fontsize=14)
    ax.set_ylabel("Number of participants", fontsize=14)
    ax.legend(fontsize=12, loc="upper left")
    ax.set_ylim(0, top * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(filename, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [saved] {os.path.basename(filename)}")

def plot_all_signals_summary_with_errors(results, filename):
    """Grouped bar chart comparing Brier/AUROC/ECE with bootstrap CIs as error bars."""
    labels  = [AUROC_LABELS.get(r["signal"], r["signal"]).replace("\n", " ") for r in results]

    briers   = [r["brier"] for r in results]
    brier_yerr = [
        [r["brier"] - r["brier_ci_low"] for r in results],
        [r["brier_ci_high"] - r["brier"] for r in results]
    ]

    aurocs   = [r["auroc"] for r in results]
    auroc_yerr = [
        [r["auroc"] - r["auroc_ci_low"] for r in results],
        [r["auroc_ci_high"] - r["auroc"] for r in results]
    ]

    eces     = [r["ece"] for r in results]
    ece_yerr = [
        [r["ece"] - r["ece_ci_low"] for r in results],
        [r["ece_ci_high"] - r["ece"] for r in results]
    ]

    x   = np.arange(len(labels))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(12, 6.5))

    eb_kwargs = dict(elinewidth=1.5, capsize=4, ecolor="#2c3e50")

    ax.bar(x - w, briers, w, yerr=brier_yerr, label="Brier Score (lower better)", color="#3498db", alpha=0.85, error_kw=eb_kwargs)
    ax.bar(x,     aurocs, w, yerr=auroc_yerr, label="AUROC (higher better)",      color="#1abc9c", alpha=0.85, error_kw=eb_kwargs)
    ax.bar(x + w, eces,   w, yerr=ece_yerr,   label="ECE (lower better)",         color="#e67e22", alpha=0.85, error_kw=eb_kwargs)

    ax.axhline(0.5,  color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0.25, color="#7f8c8d", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.text(0.02, 0.52, "AUROC chance (0.50)", transform=ax.get_yaxis_transform(),
            color="#7f8c8d", fontsize=11, ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=12, ha="center")
    ax.set_ylabel("Score", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [saved] {os.path.basename(filename)}")

def plot_auroc_only(results, filename):
    """AUROC per signal with bootstrap CIs and the chance line.

    One message per figure: which signals separate correct diagnoses from wrong
    ones, and which cannot be told apart from chance. Bars whose interval clears
    0.5 are drawn in colour, the rest in grey.
    """
    rows = [r for r in results if not np.isnan(r["auroc"])]
    if not rows:
        return

    labels  = [AUROC_LABELS.get(r["signal"], r["signal"]) for r in rows]
    vals    = [r["auroc"] for r in rows]
    yerr    = [
        [r["auroc"] - r["auroc_ci_low"]  for r in rows],
        [r["auroc_ci_high"] - r["auroc"] for r in rows],
    ]
    colours = [C_ABOVE if r["auroc_ci_low"] > 0.5 else C_CHANCE for r in rows]

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x, vals, 0.55, yerr=yerr, color=colours,
           edgecolor="white", linewidth=0.8,
           error_kw=dict(elinewidth=1.6, capsize=6, ecolor="#2c3e50"))

    ax.axhline(0.5, color=C_REF, linestyle="--", linewidth=1.4)
    ax.text(0.02, 0.515, "chance (0.50)", transform=ax.get_yaxis_transform(),
            color=C_REF, fontsize=12, ha="left", va="bottom")

    for xi, r in zip(x, rows):
        ax.text(xi, r["auroc_ci_high"] + 0.03, f"{r['auroc']:.2f}",
                ha="center", va="bottom", fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("AUROC (confidence vs correctness)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(filename, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [saved] {os.path.basename(filename)}")

def main():
    print("=" * 72)
    print("RQ3 CALIBRATION ANALYSIS V2 — Correctness Confidence & Bootstrap CIs")
    print("=" * 72)

    df = load_main_meta()
    print(f"Loaded {len(df)} subjects from main condition.\n")

    y = df["y"].values

    print("── Correct/Incorrect breakdown ──────────────────────────────")
    print(f"  Correct  : {(y == 1).sum()} / {len(y)} ({(y == 1).sum()/len(y)*100:.1f}%)")
    print(f"  Incorrect: {(y == 0).sum()} / {len(y)} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print()

    results = []
    for label, (col, title) in SIGNAL_META.items():
        y_prob = df[col].values.astype(float) if col in df.columns else np.full(len(y), np.nan)

        metrics = compute_all_metrics_with_ci(y, y_prob, label)
        results.append(metrics)

        print(f"── {label} (n={metrics['n']})")
        print(f"   Brier Score : {metrics['brier']:.4f} (95% CI: {metrics['brier_ci_low']:.4f} - {metrics['brier_ci_high']:.4f})")
        print(f"   AUROC       : {metrics['auroc']:.4f} (95% CI: {metrics['auroc_ci_low']:.4f} - {metrics['auroc_ci_high']:.4f})")
        print(f"   ECE         : {metrics['ece']:.4f} (95% CI: {metrics['ece_ci_low']:.4f} - {metrics['ece_ci_high']:.4f})")
        print(f"   Mean conf (correct) : {metrics['mean_conf_correct']:.4f}")
        print(f"   Mean conf (wrong)   : {metrics['mean_conf_wrong']:.4f}")
        print()

        safe_label = label.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
        plot_reliability_diagram(
            y, y_prob, label,
            os.path.join(FIG_DIR, f"v2_reliability_{safe_label}.png")
        )
        plot_confidence_distribution(
            y, y_prob, label,
            os.path.join(FIG_DIR, f"v2_confidence_dist_{safe_label}.png")
        )

    plot_all_signals_summary_with_errors(
        results,
        os.path.join(FIG_DIR, "v2_calibration_summary_all_signals.png")
    )

    plot_auroc_only(
        results,
        os.path.join(FIG_DIR, "v2_auroc_only.png")
    )

    summary_rows = []
    for r in results:
        summary_rows.append({
            "Signal": r["signal"],
            "N": r["n"],
            "Brier Score": f"{r['brier']:.4f} ({r['brier_ci_low']:.4f}, {r['brier_ci_high']:.4f})",
            "AUROC": f"{r['auroc']:.4f} ({r['auroc_ci_low']:.4f}, {r['auroc_ci_high']:.4f})",
            "ECE": f"{r['ece']:.4f} ({r['ece_ci_low']:.4f}, {r['ece_ci_high']:.4f})",
            "Mean Conf (Correct)": f"{r['mean_conf_correct']:.4f}" if not np.isnan(r['mean_conf_correct']) else "N/A",
            "Mean Conf (Incorrect)": f"{r['mean_conf_wrong']:.4f}" if not np.isnan(r['mean_conf_wrong']) else "N/A",
        })

    summary_df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(TAB_DIR, "v2_calibration_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[saved] v2_calibration_summary.csv")
    print("\n── Summary Table (Point Estimate and 95% Bootstrap CI) ──────")
    print(summary_df.to_string(index=False))

    print("\n── Interpretation ───────────────────────────────────────────")
    for r in results:
        if np.isnan(r["auroc"]):
            continue
        ci_str = f"95% CI: [{r['auroc_ci_low']:.3f}, {r['auroc_ci_high']:.3f}]"
        if r["auroc_ci_low"] <= 0.5 and r["auroc_ci_high"] >= 0.5:
            print(f"  [UNINFORMATIVE] {r['signal']}: AUROC={r['auroc']:.3f} ({ci_str}) — CI overlaps 0.5, no statistically significant discriminative power.")
        elif r["auroc"] < 0.70:
            print(f"  [WEAK SIGN] {r['signal']}: AUROC={r['auroc']:.3f} ({ci_str})")
        else:
            print(f"  [INFORMATIVE] {r['signal']}: AUROC={r['auroc']:.3f} ({ci_str})")

    print("\nDone. Figures in:", FIG_DIR)
    print("Tables in:", TAB_DIR)

if __name__ == "__main__":
    main()
