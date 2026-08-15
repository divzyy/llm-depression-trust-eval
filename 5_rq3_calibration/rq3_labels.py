

import os
import sys

import pandas as pd

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

LABEL_DIR = f"{DAIC}/labels"

LABEL_FILES = [
    "train_split_Depression_AVEC2017.csv",
    "dev_split_Depression_AVEC2017.csv",
]

# A PHQ-8 total of 10 or more counts as a positive depression screen.
PHQ8_CUTOFF = 10

_PID_CANDIDATES = [
    "Participant_ID", "participant_ID", "participant_id", "PID", "pid",
]
_SCORE_CANDIDATES = [
    "PHQ8_Score", "PHQ8_score", "PHQ_8Total", "PHQ8_Total", "phq8_score",
]

def _pick_column(frame, candidates, what, path):
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(
        f"Could not find a {what} column in {path}. "
        f"Columns present: {list(frame.columns)}. "
        f"Add the right name to the candidate list in rq3_labels.py."
    )

def load_phq8_totals(label_dir=LABEL_DIR):
    """Return {participant_id: PHQ-8 total} from the official label files."""
    totals = {}
    for fname in LABEL_FILES:
        path = os.path.join(label_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Label file not found: {path}. "
                f"Set LABEL_DIR in rq3_labels.py to the right directory."
            )
        labels = pd.read_csv(path)
        pid_col = _pick_column(labels, _PID_CANDIDATES, "participant id", path)
        score_col = _pick_column(labels, _SCORE_CANDIDATES, "PHQ-8 total", path)
        for pid, score in zip(labels[pid_col], labels[score_col]):
            totals[int(pid)] = float(score)
    return totals

def apply_gt_fix(df, label_dir=LABEL_DIR, verbose=True):
    """
    Overwrite df['gt_binary'] with the PHQ-8 rule (total >= PHQ8_CUTOFF).

    Returns the same DataFrame, modified in place. Prints every participant
    whose label changes, and the class counts after the fix.

    A participant with no PHQ-8 total in the label files keeps its stored
    gt_binary, and is reported as a warning.
    """
    totals = load_phq8_totals(label_dir)

    pids = df["pid"].astype(int)
    old = pd.to_numeric(df["gt_binary"], errors="coerce")

    def rule(pid):
        if pid not in totals:
            return None
        return int(totals[pid] >= PHQ8_CUTOFF)

    derived = pids.map(rule)

    missing = sorted(set(pids[derived.isna()].tolist()))
    if missing and verbose:
        print(f"  [WARN] no PHQ-8 total in the label files for pid(s) {missing}; "
              f"their stored gt_binary is kept")

    new = derived.fillna(old)
    both_known = new.notna() & old.notna()
    changed = sorted(pids[both_known & (new != old)].tolist())

    df["gt_binary"] = pd.to_numeric(new, errors="coerce")

    if verbose:
        n_dep = int((df["gt_binary"] == 1).sum())
        n_not = int((df["gt_binary"] == 0).sum())
        if changed:
            for pid in changed:
                was = int(old[pids == pid].iloc[0])
                now = int(new[pids == pid].iloc[0])
                total = totals[pid]
                print(f"  [gt fix] pid {pid}: gt_binary {was} -> {now} "
                      f"(PHQ-8 total = {total:.0f}, cut-off = {PHQ8_CUTOFF})")
        else:
            print("  [gt fix] no label changed; the stored gt_binary already "
                  "follows the PHQ-8 rule")
        print(f"  [gt fix] ground truth after fix: {n_dep} depressed, "
              f"{n_not} not depressed, {len(df)} participants")

    return df

def _standalone(condition="main"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "..", "analysis_output", "rq3",
                        f"rq3_{condition}_meta.csv")
    print("=" * 72)
    print(f"rq3_labels — dry check on condition: {condition}")
    print("=" * 72)
    print(f"Reading {path}")

    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df[pd.to_numeric(df["pid"], errors="coerce").notna()].copy()
    df["pid"] = df["pid"].astype(int)
    df["binary"] = pd.to_numeric(df["binary"], errors="coerce")

    before = pd.to_numeric(df["gt_binary"], errors="coerce")
    correct_before = int((df["binary"] == before).sum())

    apply_gt_fix(df)

    correct_after = int((df["binary"] == df["gt_binary"]).sum())
    n = len(df)
    print(f"  correct before fix: {correct_before} of {n}")
    print(f"  correct after fix : {correct_after} of {n}")

    wrong = df[df["binary"] != df["gt_binary"]].sort_values("pid")
    print(f"\n  wrong diagnoses after fix ({len(wrong)}):")
    cols = [c for c in ["pid", "binary", "gt_binary", "s1_p_depressed", "s2_conf"]
            if c in wrong.columns]
    print(wrong[cols].to_string(index=False))
    print("\nNothing was written. This is a check only.")

if __name__ == "__main__":
    _standalone(sys.argv[1] if len(sys.argv) > 1 else "main")
