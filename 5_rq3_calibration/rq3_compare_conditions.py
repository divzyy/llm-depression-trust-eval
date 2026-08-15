
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

PRETTY = {
    "main":              "severity first",
    "no_qual":           "no qualitative",
    "no_quant":          "no quantitative",
    "transcript_only":   "transcript only",
    "explanation_first": "explanation first",
}
COL_A, COL_B = "#3498db", "#9b59b6"
C_REF = "#7f8c8d"

SIGNALS = [
    ("S1 Token\nLogprob",   "s1_conf"),
    ("S2 Verbalized",       "s2_conf_norm"),
    ("S3 Sampling\nAgree",  "s3_agree_binary"),
    ("S4 Mean\nP(dep)",     "s4_conf"),
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

    df["y"] = (df["binary"] == df["gt_binary"]).astype(int)
    df["s2_conf_norm"] = df["s2_conf"] / 100.0
    df["s1_conf"] = np.where(df["binary"] == 1, df["s1_p_depressed"], 1.0 - df["s1_p_depressed"])
    df["s4_conf"] = np.where(df["binary"] == 1, df["s4_mean_p_depressed"], 1.0 - df["s4_mean_p_depressed"])
    return df

def ece(y, p, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    v = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (p >= lo) & (p <= hi) if i == nb - 1 else (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        v += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return v

def boot(y, p, fn, n=1000, seed=42):
    valid = ~np.isnan(p)
    yt, yp = y[valid], p[valid]
    if len(yt) < 5:
        return np.nan, np.nan, np.nan
    try:
        pe = fn(yt, yp)
    except Exception:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.choice(len(yt), len(yt), replace=True)
        try:
            v = fn(yt[i], yp[i])
            if not np.isnan(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return pe, np.nan, np.nan
    return pe, np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def verdict(au, lo, hi):
    if np.isnan(au):
        return "N/A"
    if lo <= 0.5 <= hi:
        return "UNINFORMATIVE"
    return "WEAK SIGN" if au < 0.70 else "INFORMATIVE"

def analyse(df):
    y = df["y"].values
    out = {}
    for label, col in SIGNALS:
        p = df[col].values.astype(float) if col in df.columns else np.full(len(y), np.nan)
        if np.isnan(p).all():
            out[label] = None
            continue
        au, au_l, au_h = boot(y, p, roc_auc_score)
        br, _, _ = boot(y, p, lambda a, b: brier_score_loss(a, b))
        ec, _, _ = boot(y, p, lambda a, b: ece(a, b))
        valid = ~np.isnan(p)
        yt, yp = y[valid], p[valid]
        cr = yp[yt == 1].mean() if (yt == 1).sum() else np.nan
        cw = yp[yt == 0].mean() if (yt == 0).sum() else np.nan
        out[label] = {"auroc": au, "lo": au_l, "hi": au_h, "brier": br, "ece": ec,
                      "conf_right": cr, "conf_wrong": cw, "gap": cr - cw,
                      "verdict": verdict(au, au_l, au_h), "n": int(valid.sum())}
    return y.mean(), int((y == 0).sum()), out

def plot_auroc(res_a, res_b, na, nb, path):
    labels = [l for l, _ in SIGNALS]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for off, res, colour, name in ((-w/2, res_a, COL_A, na), (w/2, res_b, COL_B, nb)):
        vals, errs, xs = [], [[], []], []
        for i, l in enumerate(labels):
            r = res.get(l)
            if r is None or np.isnan(r["auroc"]):
                continue
            xs.append(x[i] + off); vals.append(r["auroc"])
            errs[0].append(r["auroc"] - r["lo"]); errs[1].append(r["hi"] - r["auroc"])
        ax.bar(xs, vals, w, yerr=errs, capsize=4, color=colour, alpha=.88,
               edgecolor="white", label=name, error_kw=dict(elinewidth=1.4, ecolor="#2c3e50"))
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.axhline(0.5, color=C_REF, ls="--", lw=1.4)
    ax.text(len(labels) - 0.5, 0.515, "chance (0.50)", color=C_REF, fontsize=9, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC (confidence vs correctness)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Can the confidence signal tell right answers from wrong ones?\n"
                 "Bars are AUROC, whiskers are 95% bootstrap CIs. A CI crossing the\n"
                 "dashed line means the signal cannot be separated from chance.",
                 fontsize=10.5)
    ax.legend(fontsize=9.5, loc="upper left"); ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [saved] {os.path.basename(path)}")

def plot_confgap(res_a, res_b, na, nb, path):
    labels = [l for l, _ in SIGNALS]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for off, res, colour, name in ((-w/2, res_a, COL_A, na), (w/2, res_b, COL_B, nb)):
        xs, vals = [], []
        for i, l in enumerate(labels):
            r = res.get(l)
            if r is None or np.isnan(r["gap"]):
                continue
            xs.append(x[i] + off); vals.append(r["gap"])
        ax.bar(xs, vals, w, color=colour, alpha=.88, edgecolor="white", label=name)
        for xi, v in zip(xs, vals):
            va = "bottom" if v >= 0 else "top"
            ax.text(xi, v + (0.004 if v >= 0 else -0.004), f"{v:+.3f}",
                    ha="center", va=va, fontsize=8.5)

    ax.axhline(0, color="#2c3e50", lw=1.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean confidence when right  −  when wrong", fontsize=11)
    ax.set_title("Is the agent more confident when it is right?\n"
                 "Bars below zero mean the agent is MORE confident when it is WRONG.",
                 fontsize=10.5)
    ax.legend(fontsize=9.5); ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [saved] {os.path.basename(path)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="main")
    ap.add_argument("--b", default="explanation_first")
    args = ap.parse_args()
    a, b = args.a, args.b
    na, nb = PRETTY.get(a, a), PRETTY.get(b, b)

    print("=" * 78)
    print(f"RQ3 COMPARISON — {na} ({a})  vs  {nb} ({b})")
    print("=" * 78)

    da, db = load_condition(a), load_condition(b)
    ka = sorted(pd.to_numeric(da['k_samples'], errors='coerce').dropna().unique())
    kb = sorted(pd.to_numeric(db['k_samples'], errors='coerce').dropna().unique())
    if ka != kb:
        print(f"[WARNING] k_samples differ: {a}={ka}, {b}={kb}.")
        print("          S3 and S4 depend on the number of samples, so the comparison")
        print("          of those two signals is confounded. Re-run with matching K.\n")
    else:
        print(f"k_samples match ({ka}) — S3 and S4 are comparable.\n")

    acc_a, wrong_a, res_a = analyse(da)
    acc_b, wrong_b, res_b = analyse(db)
    print(f"{na:18s} n={len(da)}  accuracy={acc_a:.4f}  wrong={wrong_a}")
    print(f"{nb:18s} n={len(db)}  accuracy={acc_b:.4f}  wrong={wrong_b}")
    print(f"accuracy change: {acc_b - acc_a:+.4f}\n")

    hdr = f"{'signal':16s} {'condition':16s} {'AUROC [95% CI]':26s} {'verdict':14s} {'ECE':7s} {'gap':8s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for label, _ in SIGNALS:
        for name, cond, res in ((na, a, res_a), (nb, b, res_b)):
            r = res.get(label)
            flat = label.replace("\n", " ")
            if r is None:
                print(f"{flat:16s} {name:16s} {'not collected':26s}")
                continue
            ci = f"{r['auroc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
            print(f"{flat:16s} {name:16s} {ci:26s} {r['verdict']:14s} {r['ece']:<7.3f} {r['gap']:+.3f}")
            rows.append({"signal": flat, "condition": cond, "condition_label": name,
                         "n": r["n"], "accuracy": round(acc_a if cond == a else acc_b, 4),
                         "auroc": round(r["auroc"], 4),
                         "auroc_ci": f"[{r['lo']:.4f}, {r['hi']:.4f}]",
                         "verdict": r["verdict"], "brier": round(r["brier"], 4),
                         "ece": round(r["ece"], 4),
                         "conf_right": round(r["conf_right"], 4),
                         "conf_wrong": round(r["conf_wrong"], 4),
                         "conf_gap": round(r["gap"], 4)})
        print()

    print("── Figures ──")
    plot_auroc(res_a, res_b, na, nb, os.path.join(FIG_DIR, f"compare_{a}_vs_{b}_auroc.png"))
    plot_confgap(res_a, res_b, na, nb, os.path.join(FIG_DIR, f"compare_{a}_vs_{b}_confgap.png"))

    path = os.path.join(TAB_DIR, f"compare_{a}_vs_{b}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n[saved] {os.path.basename(path)}")
    print(f"Figures: {FIG_DIR}\nTables:  {TAB_DIR}")

if __name__ == "__main__":
    main()
