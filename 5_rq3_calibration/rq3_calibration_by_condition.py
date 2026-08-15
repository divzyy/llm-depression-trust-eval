

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from rq3_labels import apply_gt_fix

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.join(SCRIPT_DIR, "..")
DATA_DIR   = os.path.join(ROOT, "analysis_output", "rq3")
OUT_DIR    = os.path.join(SCRIPT_DIR, "output")
FIG_DIR    = os.path.join(OUT_DIR, "figures")
TAB_DIR    = os.path.join(OUT_DIR, "tables")
for d in (FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

C_CORRECT = "#2ecc71"
C_WRONG   = "#e74c3c"
C_CALIB   = "#9b59b6"
C_REF     = "#7f8c8d"

SIGNALS = [
    ("S1 Token Logprob",    "s1_conf"),
    ("S2 Verbalized Conf.", "s2_conf_norm"),
    ("S3 Sampling Agree",   "s3_agree_binary"),
    ("S4 Mean P(dep)",      "s4_conf"),
]

def load_condition(condition):
    path = os.path.join(DATA_DIR, f"rq3_{condition}_meta.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df[pd.to_numeric(df["pid"], errors="coerce").notna()].copy()
    df["pid"] = df["pid"].astype(int)

    for c in ["binary", "gt_binary", "s1_p_depressed",
              "s3_agree_binary", "s4_mean_p_depressed", "s2_conf"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = apply_gt_fix(df)
    
    df["correct_bool"] = df["binary"] == df["gt_binary"]
    df["y"] = df["correct_bool"].astype(int)
    df["s2_conf_norm"] = df["s2_conf"] / 100.0
    df["s1_conf"] = np.where(df["binary"] == 1, df["s1_p_depressed"], 1.0 - df["s1_p_depressed"])
    df["s4_conf"] = np.where(df["binary"] == 1, df["s4_mean_p_depressed"], 1.0 - df["s4_mean_p_depressed"])
    return df

def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    val = 0.0
    bin_data = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            bin_data.append((lo, hi, 0, np.nan, np.nan))
            continue
        acc, conf = y_true[mask].mean(), y_prob[mask].mean()
        val += mask.sum() / len(y_true) * abs(acc - conf)
        bin_data.append((lo, hi, int(mask.sum()), acc, conf))
    return val, bin_data

def bootstrap_metric(y_true, y_prob, fn, n_bootstrap=1000, rng_seed=42):
    valid = ~np.isnan(y_prob)
    yt, yp = y_true[valid], y_prob[valid]
    if len(yt) < 5:
        return np.nan, np.nan, np.nan
    try:
        pe = fn(yt, yp)
    except Exception:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(rng_seed)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(yt), len(yt), replace=True)
        try:
            v = fn(yt[idx], yp[idx])
            if not np.isnan(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return pe, np.nan, np.nan
    return pe, np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def metrics_for_signal(y, p, label):
    br, br_l, br_h = bootstrap_metric(y, p, lambda a, b: brier_score_loss(a, b))
    au, au_l, au_h = bootstrap_metric(y, p, lambda a, b: roc_auc_score(a, b))
    ec, ec_l, ec_h = bootstrap_metric(y, p, lambda a, b: ece(a, b)[0])
    valid = ~np.isnan(p)
    yt, yp = y[valid], p[valid]
    c_right = yp[yt == 1].mean() if (yt == 1).sum() else np.nan
    c_wrong = yp[yt == 0].mean() if (yt == 0).sum() else np.nan
    return {
        "signal": label, "n": int(valid.sum()),
        "brier": br, "brier_lo": br_l, "brier_hi": br_h,
        "auroc": au, "auroc_lo": au_l, "auroc_hi": au_h,
        "ece": ec, "ece_lo": ec_l, "ece_hi": ec_h,
        "conf_right": c_right, "conf_wrong": c_wrong,
        "conf_gap": c_right - c_wrong if not (np.isnan(c_right) or np.isnan(c_wrong)) else np.nan,
    }

def verdict(r):
    """Same rule the v2 script uses, so the wording stays consistent."""
    if np.isnan(r["auroc"]):
        return "N/A"
    if r["auroc_lo"] <= 0.5 <= r["auroc_hi"]:
        return "UNINFORMATIVE"
    return "WEAK SIGN" if r["auroc"] < 0.70 else "INFORMATIVE"

def plot_reliability(y, p, label, cond, path, n_bins=10):
    valid = ~np.isnan(p)
    yt, yp = y[valid], p[valid]
    if len(yt) == 0:
        return
    val, bin_data = ece(yt, yp, n_bins)
    xs, ys, ns = [], [], []
    for lo, hi, cnt, acc, conf in bin_data:
        if cnt > 0 and not np.isnan(acc):
            xs.append(conf); ys.append(acc); ns.append(cnt)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color=C_REF, lw=1.5, label="Perfect calibration")
    if xs:
        ax.plot(xs, ys, "o-", color=C_CALIB, ms=8, lw=2, label="Calibration curve")
        for n, x, yv in zip(ns, xs, ys):
            ax.annotate(f"n={n}", (x, yv), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color="#34495e",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="#bdc3c7"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence (correctness-oriented)", fontsize=11)
    ax.set_ylabel("Observed accuracy", fontsize=11)
    ax.set_title(f"Reliability — {label}\n{cond}, ECE = {val:.4f}", fontsize=11)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [saved] {os.path.basename(path)}")

def plot_confdist(y, p, label, cond, path):
    valid = ~np.isnan(p)
    yt, yp = y[valid], p[valid]
    if len(yt) == 0:
        return
    right, wrong = yp[yt == 1], yp[yt == 0]
    nb = 10
    edges = np.linspace(0, 1, nb + 1)
    rc = np.zeros(nb, int); wc = np.zeros(nb, int)
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        sel = (lambda a: (a >= lo) & (a <= hi)) if i == nb - 1 else (lambda a: (a >= lo) & (a < hi))
        rc[i] = int(sel(right).sum()); wc[i] = int(sel(wrong).sum())
    keep = np.where((rc + wc) > 0)[0]
    if len(keep) == 0:
        return
    labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in keep]
    x = np.arange(len(keep)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(keep) * 1.4), 5))
    b1 = ax.bar(x - w/2, rc[keep], w, color=C_CORRECT, edgecolor="white", label=f"Correct (n={len(right)})")
    b2 = ax.bar(x + w/2, wc[keep], w, color=C_WRONG,   edgecolor="white", label=f"Incorrect (n={len(wrong)})")
    for bars in (b1, b2):
        for b in bars:
            if b.get_height() > 0:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                        str(int(b.get_height())), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_xlabel("Confidence (bin)", fontsize=11); ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Confidence distribution — {label}\n{cond}", fontsize=11)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [saved] {os.path.basename(path)}")

def plot_summary(results, cond, path):
    labels = [r["signal"] for r in results]
    x = np.arange(len(labels)); w = 0.25
    def err(key):
        return [[r[key] - r[key + "_lo"] for r in results],
                [r[key + "_hi"] - r[key] for r in results]]
    fig, ax = plt.subplots(figsize=(10, 6))
    kw = dict(elinewidth=1.5, capsize=4, ecolor="#2c3e50")
    ax.bar(x - w, [r["brier"] for r in results], w, yerr=err("brier"),
           label="Brier (lower better)", color="#3498db", alpha=.85, error_kw=kw)
    ax.bar(x,     [r["auroc"] for r in results], w, yerr=err("auroc"),
           label="AUROC (higher better)", color="#1abc9c", alpha=.85, error_kw=kw)
    ax.bar(x + w, [r["ece"] for r in results],   w, yerr=err("ece"),
           label="ECE (lower better)", color="#e67e22", alpha=.85, error_kw=kw)
    ax.axhline(0.5, color=C_REF, ls="--", lw=1.0, alpha=.7)
    ax.text(-0.45, 0.52, "AUROC chance (0.50)", color=C_REF, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Calibration metrics with 95% bootstrap CIs — {cond}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right"); ax.set_ylim(0, 1.1); ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [saved] {os.path.basename(path)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["main", "no_qual", "no_quant", "transcript_only", "explanation_first"])
    args = ap.parse_args()
    cond = args.condition

    print("=" * 72)
    print(f"RQ3 CALIBRATION — condition: {cond}")
    print("=" * 72)

    df = load_condition(cond)
    y = df["y"].values
    print(f"Loaded {len(df)} subjects | correct {int((y==1).sum())} | wrong {int((y==0).sum())} "
          f"| accuracy {y.mean():.4f}")
    ks = sorted(pd.to_numeric(df['k_samples'], errors='coerce').dropna().unique())
    print(f"k_samples present: {ks}\n")

    results = []
    for label, col in SIGNALS:
        p = df[col].values.astype(float) if col in df.columns else np.full(len(y), np.nan)
        if np.isnan(p).all():
            print(f"── {label}: not collected in this condition, skipped\n")
            continue
        r = metrics_for_signal(y, p, label)
        r["verdict"] = verdict(r)
        results.append(r)
        print(f"── {label} (n={r['n']})  [{r['verdict']}]")
        print(f"   Brier {r['brier']:.4f} [{r['brier_lo']:.4f}, {r['brier_hi']:.4f}]")
        print(f"   AUROC {r['auroc']:.4f} [{r['auroc_lo']:.4f}, {r['auroc_hi']:.4f}]")
        print(f"   ECE   {r['ece']:.4f} [{r['ece_lo']:.4f}, {r['ece_hi']:.4f}]")
        print(f"   conf when right {r['conf_right']:.4f} | when wrong {r['conf_wrong']:.4f} "
              f"| gap {r['conf_gap']:+.4f}\n")

        safe = label.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
        plot_reliability(y, p, label, cond, os.path.join(FIG_DIR, f"calib_{cond}_reliability_{safe}.png"))
        plot_confdist(y, p, label, cond, os.path.join(FIG_DIR, f"calib_{cond}_confdist_{safe}.png"))

    if not results:
        print("[ERROR] no signals available for this condition.")
        return

    plot_summary(results, cond, os.path.join(FIG_DIR, f"calib_{cond}_summary_all_signals.png"))

    out = pd.DataFrame([{
        "condition": cond, "signal": r["signal"], "n": r["n"], "verdict": r["verdict"],
        "accuracy": round(float(y.mean()), 4),
        "brier": round(r["brier"], 4), "brier_ci": f"[{r['brier_lo']:.4f}, {r['brier_hi']:.4f}]",
        "auroc": round(r["auroc"], 4), "auroc_ci": f"[{r['auroc_lo']:.4f}, {r['auroc_hi']:.4f}]",
        "ece": round(r["ece"], 4),     "ece_ci":   f"[{r['ece_lo']:.4f}, {r['ece_hi']:.4f}]",
        "conf_right": round(r["conf_right"], 4), "conf_wrong": round(r["conf_wrong"], 4),
        "conf_gap": round(r["conf_gap"], 4),
    } for r in results])
    path = os.path.join(TAB_DIR, f"calib_{cond}_summary.csv")
    out.to_csv(path, index=False)
    print(f"\n[saved] {os.path.basename(path)}")
    print(out.to_string(index=False))
    print(f"\nFigures: {FIG_DIR}\nTables:  {TAB_DIR}")

if __name__ == "__main__":
    main()
