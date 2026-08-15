#!/usr/bin/env python3
"""
verify_fewshot_split.py

Checks whether the few-shot knowledge base leakage (participants 339 and 345)
was actually fixed, and whether the few-shot numbers in the thesis came from
the clean knowledge base.

Run on ALICE (login node is fine), in the aipsy conda env:

    conda activate aipsy
    python verify_fewshot_split.py

Four independent checks. Each prints PASS / FAIL / SKIP and says why.
Nothing is written or modified.
"""

import os
import sys
import json
import pickle
import re
from datetime import datetime

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))


PICKLE      = f"{ROOT}/agents/chunk_8_step_2_participant_embedded_transcripts.pkl"
MANIFEST_JS = f"{ROOT}/agents/split_manifest_58_43_41.json"
MANIFEST_PY = f"{ROOT}/split_manifest.py"
FEWSHOT_DIR = f"{ROOT}/analysis_output/quan_gemma_few_shot"
JSONL       = f"{FEWSHOT_DIR}/ids_test_chunk_8_step_2_dim_4096_examples_2_embedding_results_analysis_1.jsonl"
LOGFILE     = f"{FEWSHOT_DIR}/embedding_log_file.txt"
META_CSV    = f"{ROOT}/analysis_output/qual/meta_review_fewshot_test_v2.csv"

LINE = "=" * 72

def mtime(path):
    """File modification time, or None if the file is missing."""
    if not os.path.exists(path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(path))

def head(title):
    print(f"\n{LINE}\n{title}\n{LINE}")

def get_test_ids():
    """
    Test participant IDs. Prefer split_manifest.py (the source of truth used by
    generate_pickle.py). Fall back to the participants present in the few-shot
    test JSONL, which is by construction the test set.
    """
    if os.path.exists(MANIFEST_PY):
        sys.path.insert(0, os.path.dirname(MANIFEST_PY))
        try:
            from split_manifest import TEST_IDS
            return set(int(x) for x in TEST_IDS), "split_manifest.py"
        except Exception as e:
            print(f"  note: could not import split_manifest.py ({e})")
    if os.path.exists(JSONL):
        ids = set()
        with open(JSONL) as f:
            for line in f:
                if line.strip():
                    ids.add(int(json.loads(line)["participant_id"]))
        return ids, "few-shot test JSONL"
    return set(), None

# CHECK 1 - Does the knowledge base on disk contain any test participant?
def check_1_pickle(test_ids):
    head("CHECK 1  Knowledge base contents (the leakage check)")

    if not os.path.exists(PICKLE):
        print(f"SKIP  pickle not found at:\n      {PICKLE}")
        return None

    with open(PICKLE, "rb") as f:
        kb = pickle.load(f)

    kb_ids = set()
    for k in kb.keys():
        try:
            kb_ids.add(int(k))
        except (ValueError, TypeError):
            pass

    overlap = sorted(kb_ids & test_ids)

    print(f"pickle           : {PICKLE}")
    print(f"built            : {mtime(PICKLE)}")
    print(f"participants     : {len(kb_ids)}   (clean = 58)")
    print(f"339 present      : {339 in kb_ids}   (clean = False)")
    print(f"345 present      : {345 in kb_ids}   (clean = False)")
    print(f"test IDs inside  : {overlap if overlap else 'none'}")

    ok = (len(kb_ids) == 58) and not overlap
    print(f"\n{'PASS  knowledge base is clean.' if ok else 'FAIL  knowledge base still contains test participants.'}")
    if not ok and len(kb_ids) != 58:
        print(f"      Size is {len(kb_ids)}, not 58. This may be the old pickle.")
    return ok

# CHECK 2 - Did the few-shot outputs come from the clean pickle?
def check_2_provenance():
    head("CHECK 2  Provenance: were the few-shot outputs made AFTER the rebuild?")

    t_pickle = mtime(PICKLE)
    t_jsonl = mtime(JSONL)
    t_meta = mtime(META_CSV)

    print(f"pickle built     : {t_pickle}")
    print(f"few-shot JSONL   : {t_jsonl}")
    print(f"meta-review CSV  : {t_meta}")

    # Internal per-participant timestamps are more reliable than mtime,
    # because copying a file changes mtime but not its contents.
    internal = []
    if os.path.exists(JSONL):
        with open(JSONL) as f:
            for line in f:
                if line.strip():
                    ts = json.loads(line).get("timestamp")
                    if ts:
                        internal.append(ts)
    if internal:
        print(f"\nJSONL internal timestamps (written at run time):")
        print(f"  earliest       : {min(internal)}")
        print(f"  latest         : {max(internal)}")
        print(f"  records        : {len(internal)}")

    if t_pickle is None or not internal:
        print("\nSKIP  need both the pickle and the JSONL to compare.")
        return None

    latest_run = datetime.fromisoformat(max(internal))
    ok = latest_run > t_pickle
    print()
    if ok:
        print("PASS  the few-shot run postdates the pickle, so it used the clean base.")
    else:
        print("FAIL  the few-shot run PREDATES the pickle.")
        print("      These scores came from the old knowledge base. Re-run:")
        print("        sbatch job3fewmeta.sh")
    return ok

# CHECK 3 - Which participants were actually retrieved as references?
def check_3_retrieval_log(test_ids):
    head("CHECK 3  Retrieval log: was any test participant ever retrieved?")

    if not os.path.exists(LOGFILE):
        print(f"SKIP  log not found at:\n      {LOGFILE}")
        return None

    # find_similar_chunks logs, via log_message:
    #   [YYYY-MM-DD HH:MM:SS]   Rank 1: participant=123, chunk_id=45, similarity=0.7342
    pat = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*Rank\s*\d+:\s*participant=(\d+)"
    )

    t_pickle = mtime(PICKLE)
    all_hits, recent_hits = [], []

    with open(LOGFILE, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            when = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            pid = int(m.group(2))
            all_hits.append((when, pid))
            if t_pickle and when > t_pickle:
                recent_hits.append((when, pid))

    if not all_hits:
        print("SKIP  no 'Rank N: participant=' lines found.")
        print("      The log may predate that logging, or use another format.")
        return None

    def report(hits, label):
        pids = set(p for _, p in hits)
        leaked = sorted(pids & test_ids)
        print(f"\n{label}")
        print(f"  retrievals logged     : {len(hits)}")
        print(f"  distinct participants : {len(pids)}")
        print(f"  window                : {min(w for w,_ in hits)}  ->  {max(w for w,_ in hits)}")
        print(f"  TEST participants retrieved: {leaked if leaked else 'none'}")
        return not leaked

    ok_all = report(all_hits, "Whole log (all runs, including any pre-fix run):")

    ok = ok_all
    if t_pickle and recent_hits:
        ok = report(recent_hits, "Only retrievals AFTER the pickle was rebuilt:")
    elif t_pickle:
        print("\n  No retrievals logged after the pickle was rebuilt.")
        print("  That points to the few-shot step not having been re-run.")
        ok = False

    print()
    print("PASS  no test participant was retrieved as a reference." if ok
          else "FAIL  a test participant was retrieved as a reference example.")
    return ok

# CHECK 4 - How was the split recovered? (settles the Section 7.1 wording)
def check_4_split_provenance():
    head("CHECK 4  Split provenance (for the Section 7.1 wording)")

    if os.path.exists(MANIFEST_JS):
        with open(MANIFEST_JS) as f:
            man = json.load(f)
        print(f"manifest written : {mtime(MANIFEST_JS)}")
        for key in ("train_ids", "val_ids", "validation_ids", "test_ids"):
            if key in man:
                print(f"  {key:16s}: {len(man[key])}")
        for key, val in man.items():
            if isinstance(val, str):
                print(f"  {key:16s}: {val}")
    else:
        print(f"note: {MANIFEST_JS} not found.")

    if os.path.exists(MANIFEST_PY):
        print(f"\n--- head of split_manifest.py (its own account of where IDs came from) ---")
        with open(MANIFEST_PY, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 40:
                    print("  ...")
                    break
                print("  " + line.rstrip())
    else:
        print(f"\nnote: {MANIFEST_PY} not found.")

    print("\nREAD THIS YOURSELF. The question to answer:")
    print("  Were the 43 validation IDs recovered from the authors' released")
    print("  files, or only inferred as 'everything left over'?")
    print("  If recovered  -> Section 7.1's 'could not be recovered' is wrong.")
    print("  If inferred   -> keep 7.1, but say the TEST split WAS recovered.")
    return None

def main():
    print(LINE)
    print("Few-shot knowledge base verification")
    print(f"run at: {datetime.now()}")
    print(LINE)

    test_ids, src = get_test_ids()
    if not test_ids:
        sys.exit("ERROR: could not determine the test IDs. Check the paths at the top.")
    print(f"\ntest participants: {len(test_ids)}  (source: {src})")
    print(f"339 in test set: {339 in test_ids}   345 in test set: {345 in test_ids}")

    results = {
        "1 knowledge base clean": check_1_pickle(test_ids),
        "2 outputs postdate fix": check_2_provenance(),
        "3 no test retrieved":    check_3_retrieval_log(test_ids),
    }
    check_4_split_provenance()

    head("SUMMARY")
    for name, ok in results.items():
        print(f"  {name:26s}: {'PASS' if ok else 'FAIL' if ok is False else 'SKIP'}")
    print("\nChecks 1 and 3 test the leakage itself.")
    print("Check 2 tests whether the thesis numbers came from the clean run.")

if __name__ == "__main__":
    main()
