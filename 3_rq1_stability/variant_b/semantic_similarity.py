

import os
import re
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

ALL_TEST_IDS = [
    436, 370, 428, 423, 487, 357, 383, 393, 379, 485,
    430, 489, 447, 445, 468, 451, 386, 375, 409, 455,
    413, 417, 449, 456, 385, 362, 422, 390, 459, 484,
    472, 316, 330, 338, 344, 377, 427, 319, 389, 367,
    441
]

_MARKER_PATTERN = re.compile(r'<[^>]+>|[A-Z][A-Z_]*_\d+')

SHORT_TURN_THRESHOLD = 15  

def is_trivial_turn(text: str) -> bool:
    """
    Returns True if the turn was returned unchanged by the paraphrase script
    (too short, or pure markers). These are excluded from similarity scoring.
    """
    text = text.strip()
    if len(text) <= SHORT_TURN_THRESHOLD:
        return True
    stripped = _MARKER_PATTERN.sub('', text).strip()
    if len(stripped) <= 3:
        return True
    return False

def clean_for_embedding(text: str) -> str:
    """
    Remove DAIC-WOZ annotation markers before computing embeddings.
    Markers like <sigh>, <laughter> are not natural language and would
    confuse the sentence encoder.
    """
    cleaned = _MARKER_PATTERN.sub('', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Variant B semantic similarity validation using RoBERTa-large. "
            "Compares original vs paraphrased participant turns."
        )
    )
    parser.add_argument(
        "--orig_dir",
        type=str,
        default=f"{DAIC}/transcripts",
        help="Directory containing original transcript CSV files"
    )
    parser.add_argument(
        "--varb_dir",
        type=str,
        default=f"{ROOT}/rq1_perturbations/variant_b",
        help="Directory containing Variant B transcript CSV files"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.expanduser("~/logs/variant_b"),
        help="Directory to write similarity results"
    )
    parser.add_argument(
        "--test_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional: run on specific participant IDs only. Example: test_ids 436 370"
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    if args.test_ids is not None:
        subject_ids = [i for i in args.test_ids if i in ALL_TEST_IDS]
        print(f"TEST MODE: running on {len(subject_ids)} subject(s): {subject_ids}")
    else:
        subject_ids = ALL_TEST_IDS
        print(f"FULL MODE: running on all {len(subject_ids)} test subjects")

    print("\nLoading RoBERTa-large sentence encoder...")
    print("(Model: sentence-transformers/all-roberta-large-v1)")
    print("(Justified by Yang et al. 2020 — best clinical STS performance)")
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    print("Model loaded.\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    turn_rows = []
    participant_rows = []
    skipped = []

    print(f"{'='*65}")
    print(f"{'Participant':<15} {'Turns':>8} {'Excluded':>10} {'Mean Sim':>10} {'Std':>8} {'Min':>8}")
    print(f"{'='*65}")

    for participant_id in subject_ids:
        orig_path = os.path.join(args.orig_dir, f"{participant_id}_TRANSCRIPT.csv")
        varb_path = os.path.join(args.varb_dir, f"{participant_id}_TRANSCRIPT.csv")

        if not os.path.exists(orig_path):
            print(f"  {participant_id}: original transcript not found — skipping")
            skipped.append(participant_id)
            continue
        if not os.path.exists(varb_path):
            print(f"  {participant_id}: variant B transcript not found — skipping")
            skipped.append(participant_id)
            continue

        orig_df = pd.read_csv(orig_path, sep='\t')
        varb_df = pd.read_csv(varb_path, sep='\t')

        orig_df['speaker'] = orig_df['speaker'].fillna('').astype(str)
        varb_df['speaker'] = varb_df['speaker'].fillna('').astype(str)
        orig_df['value']   = orig_df['value'].fillna('').astype(str)
        varb_df['value']   = varb_df['value'].fillna('').astype(str)

        orig_part = orig_df[orig_df['speaker'].str.strip().str.lower() == 'participant'].reset_index(drop=True)
        varb_part = varb_df[varb_df['speaker'].str.strip().str.lower() == 'participant'].reset_index(drop=True)

        if len(orig_part) != len(varb_part):
            print(f"  {participant_id}: row count mismatch ({len(orig_part)} vs {len(varb_part)}) — skipping")
            skipped.append(participant_id)
            continue

        orig_texts = []
        varb_texts = []
        excluded   = 0

        for i in range(len(orig_part)):
            orig_text = orig_part.at[i, 'value']
            varb_text = varb_part.at[i, 'value']

            if is_trivial_turn(orig_text):
                excluded += 1
                continue

            orig_clean = clean_for_embedding(orig_text)
            varb_clean = clean_for_embedding(varb_text)

            if not orig_clean or not varb_clean:
                excluded += 1
                continue

            orig_texts.append((i, orig_text, orig_clean))
            varb_texts.append((i, varb_text, varb_clean))

        if not orig_texts:
            print(f"  {participant_id}: no scoreable turns after filtering — skipping")
            skipped.append(participant_id)
            continue

        orig_sentences = [t[2] for t in orig_texts]
        varb_sentences = [t[2] for t in varb_texts]

        orig_embeddings = model.encode(orig_sentences, show_progress_bar=False)
        varb_embeddings = model.encode(varb_sentences, show_progress_bar=False)

        similarities = []
        for j in range(len(orig_embeddings)):
            sim = cosine_similarity(
                orig_embeddings[j].reshape(1, -1),
                varb_embeddings[j].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))

            turn_rows.append({
                "participant_id": participant_id,
                "turn_index":     orig_texts[j][0],
                "original":       orig_texts[j][1],
                "paraphrased":    varb_texts[j][1],
                "orig_clean":     orig_texts[j][2],
                "varb_clean":     varb_texts[j][2],
                "cosine_sim":     round(float(sim), 4)
            })

        mean_sim = float(np.mean(similarities))
        std_sim  = float(np.std(similarities))
        min_sim  = float(np.min(similarities))
        max_sim  = float(np.max(similarities))

        participant_rows.append({
            "participant_id":   participant_id,
            "total_turns":      len(orig_part),
            "scored_turns":     len(similarities),
            "excluded_turns":   excluded,
            "mean_cosine_sim":  round(mean_sim, 4),
            "std_cosine_sim":   round(std_sim,  4),
            "min_cosine_sim":   round(min_sim,  4),
            "max_cosine_sim":   round(max_sim,  4),
        })

        print(f"  {participant_id:<13} {len(orig_part):>8} {excluded:>10} {mean_sim:>10.4f} {std_sim:>8.4f} {min_sim:>8.4f}")

    print(f"{'='*65}")

    if not participant_rows:
        print("No participants processed. Check paths and IDs.")
        return

    turn_df = pd.DataFrame(turn_rows)
    turn_path = os.path.join(args.log_dir, f"per_turn_similarity_{timestamp}.csv")
    turn_df.to_csv(turn_path, index=False)

    part_df = pd.DataFrame(participant_rows)
    part_path = os.path.join(args.log_dir, f"per_participant_similarity_{timestamp}.csv")
    part_df.to_csv(part_path, index=False)

    all_means = [r["mean_cosine_sim"] for r in participant_rows]
    all_turns = [r for r in turn_rows]
    all_sims  = [r["cosine_sim"] for r in all_turns]

    overall_mean = round(float(np.mean(all_means)),  4)
    overall_std  = round(float(np.std(all_means)),   4)
    overall_min  = round(float(np.min(all_sims)),    4)
    overall_max  = round(float(np.max(all_sims)),    4)


    LOW_SIM_THRESHOLD = 0.75
    low_sim_turns = [r for r in all_turns if r["cosine_sim"] < LOW_SIM_THRESHOLD]

    summary_path = os.path.join(args.log_dir, f"overall_similarity_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Variant B Semantic Similarity Summary\n")
        f.write(f"Model: sentence-transformers/all-roberta-large-v1\n")
        f.write(f"Reference: Yang et al. (2020) JMIR Med Inform\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\n{'='*50}\n")
        f.write(f"Participants processed : {len(participant_rows)}\n")
        f.write(f"Participants skipped   : {len(skipped)}\n")
        f.write(f"Total turns scored     : {len(all_sims)}\n")
        f.write(f"\nOVERALL COSINE SIMILARITY (turn-level)\n")
        f.write(f"  Mean : {overall_mean}\n")
        f.write(f"  Std  : {overall_std}\n")
        f.write(f"  Min  : {overall_min}\n")
        f.write(f"  Max  : {overall_max}\n")
        f.write(f"\nPER-PARTICIPANT MEAN SIMILARITY\n")
        f.write(f"  Mean of means : {overall_mean}\n")
        f.write(f"  Std of means  : {overall_std}\n")
        f.write(f"  Min mean      : {round(float(np.min(all_means)), 4)}\n")
        f.write(f"  Max mean      : {round(float(np.max(all_means)), 4)}\n")
        f.write(f"\nLOW SIMILARITY TURNS (cosine < {LOW_SIM_THRESHOLD})\n")
        f.write(f"  Count: {len(low_sim_turns)}\n")
        for r in sorted(low_sim_turns, key=lambda x: x["cosine_sim"]):
            f.write(f"\n  Participant {r['participant_id']} | Turn {r['turn_index']} | Sim={r['cosine_sim']}\n")
            f.write(f"    ORIG: {r['original'][:120]}\n")
            f.write(f"    VARB: {r['paraphrased'][:120]}\n")

    print(f"\nOVERALL RESULTS")
    print(f"  Mean cosine similarity : {overall_mean}")
    print(f"  Std                    : {overall_std}")
    print(f"  Min (turn-level)       : {overall_min}")
    print(f"  Max (turn-level)       : {overall_max}")
    print(f"  Low similarity turns   : {len(low_sim_turns)} (below {LOW_SIM_THRESHOLD})")
    print(f"\nFiles written:")
    print(f"  Turn-level CSV        : {turn_path}")
    print(f"  Participant CSV       : {part_path}")
    print(f"  Overall summary       : {summary_path}")

if __name__ == "__main__":
    main()
