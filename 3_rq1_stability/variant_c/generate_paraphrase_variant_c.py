

import os
import re
import csv
import argparse
import pandas as pd
from datetime import datetime

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))


ALL_TEST_IDS = [
    316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383,
    385, 386, 389, 390, 393, 409, 413, 417, 422, 423, 427, 428, 430,
    436, 441, 445, 447, 449, 451, 455, 456, 459, 468, 472, 484, 485,
    487, 489
]

NORMALIZATION_RULES = [

    ("pretty much every day",    "nearly every day"),
    ("basically every day",      "nearly every day"),
    ("almost every day",         "nearly every day"),
    ("pretty much all the time", "nearly every day"),
    ("every day",                "nearly every day"),
    ("everyday",                 "nearly every day"),
    ("all the time",             "nearly every day"),
    ("all day",                  "nearly every day"),
    ("24/7",                     "nearly every day"),
    ("constantly",               "nearly every day"),
    ("nonstop",                  "nearly every day"),
    ("continuously",             "nearly every day"),
    ("day after day",            "nearly every day"),

    ("most of the time",         "more than half the days"),
    ("most days",                "more than half the days"),
    ("pretty often",             "more than half the days"),
    ("quite often",              "more than half the days"),
    ("fairly often",             "more than half the days"),
    ("a lot of the time",        "more than half the days"),
    ("much of the time",         "more than half the days"),
    ("many days",                "more than half the days"),
    ("many times",               "more than half the days"),
    ("frequently",               "more than half the days"),
    ("regularly",                "more than half the days"),

    ("once in a while",          "several days"),
    ("every so often",           "several days"),
    ("from time to time",        "several days"),
    ("here and there",           "several days"),
    ("now and then",             "several days"),
    ("a couple of times",        "several days"),
    ("a few times",              "several days"),
    ("a few days",               "several days"),
    ("once or twice",            "several days"),
    ("not that often",           "several days"),
    ("not very often",           "several days"),
    ("not too often",            "several days"),
    ("some days",                "several days"),
    ("occasionally",             "several days"),
    ("sometimes",                "several days"),

    ("rarely if ever",           "not at all"),
    ("not particularly",         "not at all"),
    ("haven't really",           "not at all"),
    ("not really",               "not at all"),
    ("not once",                 "not at all"),
    ("zero times",               "not at all"),
]

def build_combined_pattern(rules):
    """
    Build one combined regex pattern from all rules, sorted longest-first.
    Applied once per utterance -- prevents double-replacement.
    """
    replacement_map = {phrase: rep for phrase, rep in rules}
    sorted_phrases = sorted(replacement_map.keys(), key=len, reverse=True)
    escaped = [re.escape(p) for p in sorted_phrases]
    alternation = '|'.join(escaped)
    pattern = re.compile(
        r'(?<![A-Za-z0-9])(' + alternation + r')(?![A-Za-z0-9])',
        re.IGNORECASE
    )
    return pattern, replacement_map

COMBINED_PATTERN, REPLACEMENT_MAP = build_combined_pattern(NORMALIZATION_RULES)

def normalize_utterance(text, pattern, replacement_map):
    """
    Apply all normalization rules to a single utterance string in one pass.

    Returns
    -------
    result : str
        Normalized utterance.
    substitutions : list of dict
        Keys: 'original', 'replacement', 'context'.
    """
    substitutions = []

    def replacer(m):
        matched = m.group(0)
        canonical = replacement_map[matched.lower()]
        substitutions.append({
            "original": matched,
            "replacement": canonical,
            "context": text[max(0, m.start() - 25): m.end() + 25]
        })
        return canonical

    result = pattern.sub(replacer, text)
    return result, substitutions

def process_transcript(transcript_path, pattern, replacement_map):
    """
    Load a transcript CSV, apply normalization to participant turns only.

    Returns
    -------
    modified_df : pd.DataFrame
    stats : dict
    detailed_log : list of dict
    """
    df = pd.read_csv(transcript_path, sep='\t')
    df['speaker'] = df['speaker'].fillna('Unknown').astype(str)
    df['value'] = df['value'].fillna('').astype(str)

    total_subs = 0
    participant_turns_modified = 0
    participant_turns_total = 0
    detailed_log = []
    modified_values = []

    for idx, row in df.iterrows():
        speaker = row['speaker'].strip()
        value = row['value']

        if speaker == 'Participant':
            participant_turns_total += 1
            new_value, subs = normalize_utterance(value, pattern, replacement_map)
            modified_values.append(new_value)

            if subs:
                participant_turns_modified += 1
                total_subs += len(subs)
                for s in subs:
                    detailed_log.append({
                        "row_index": idx,
                        "original_text": value,
                        "original_match": s["original"],
                        "replacement": s["replacement"],
                        "context": s["context"]
                    })
        else:
            modified_values.append(value)

    df['value'] = modified_values

    stats = {
        "participant_turns_total": participant_turns_total,
        "participant_turns_modified": participant_turns_modified,
        "total_substitutions": total_subs,
        "substitution_rate": (
            round(participant_turns_modified / participant_turns_total, 4)
            if participant_turns_total > 0 else 0.0
        )
    }

    return df, stats, detailed_log

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Variant C: PHQ-8 frequency anchor normalization. "
            "Only participant turns are modified. No GPU required."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=f"{DAIC}/transcripts",
        help="Directory containing original transcript CSV files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{ROOT}/rq1_perturbations/variant_c",
        help="Directory to write variant C transcript CSV files"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.expanduser("~/logs/variant_c"),
        help="Directory to write summary CSV and detailed log"
    )
    parser.add_argument(
        "--test_ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional: run on specific participant IDs only (for testing). "
            "Example: --test_ids 316 385 451. "
            "If not provided, runs on all 41 test subjects."
        )
    )
    args = parser.parse_args()

    if args.test_ids is not None:
        invalid = [i for i in args.test_ids if i not in ALL_TEST_IDS]
        if invalid:
            print(f"WARNING: These IDs are not in the 41 test subjects and will be skipped: {invalid}")
        subject_ids = [i for i in args.test_ids if i in ALL_TEST_IDS]
        print(f"TEST MODE: running on {len(subject_ids)} subject(s): {subject_ids}")
    else:
        subject_ids = ALL_TEST_IDS
        print(f"FULL MODE: running on all {len(subject_ids)} test subjects")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.log_dir, f"variant_c_summary_{timestamp}.csv")
    detail_log_path = os.path.join(args.log_dir, f"variant_c_detailed_log_{timestamp}.txt")

    summary_rows = []
    all_detail_lines = []

    print(f"\n=== Variant C: PHQ-8 Frequency Anchor Normalization ===")
    print(f"Rules defined:       {len(NORMALIZATION_RULES)}")
    print(f"Subjects to process: {len(subject_ids)}")
    print(f"Output dir:          {args.output_dir}")
    print(f"Log dir:             {args.log_dir}")
    print(f"Timestamp:           {timestamp}\n")

    for participant_id in subject_ids:
        transcript_path = os.path.join(
            args.data_dir, f"{participant_id}_TRANSCRIPT.csv"
        )

        if not os.path.exists(transcript_path):
            print(f"[SKIP] {participant_id}: not found at {transcript_path}")
            summary_rows.append({
                "participant_id": participant_id,
                "status": "skipped",
                "participant_turns_total": None,
                "participant_turns_modified": None,
                "total_substitutions": None,
                "substitution_rate": None
            })
            continue

        print(f"[PROCESSING] {participant_id} ...", end=" ", flush=True)

        try:
            modified_df, stats, detailed_log = process_transcript(
                transcript_path, COMBINED_PATTERN, REPLACEMENT_MAP
            )

            out_path = os.path.join(
                args.output_dir, f"{participant_id}_TRANSCRIPT_varC.csv"
            )
            modified_df.to_csv(out_path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)

            print(
                f"OK | turns modified: {stats['participant_turns_modified']}"
                f"/{stats['participant_turns_total']} "
                f"| substitutions: {stats['total_substitutions']}"
            )

            summary_rows.append({
                "participant_id": participant_id,
                "status": "ok",
                **stats
            })

            if detailed_log:
                all_detail_lines.append(f"\n{'='*60}")
                all_detail_lines.append(f"Participant: {participant_id}")
                all_detail_lines.append(f"{'='*60}")
                for entry in detailed_log:
                    all_detail_lines.append(
                        f"  Row {entry['row_index']}: "
                        f'"{entry["original_match"]}" -> "{entry["replacement"]}"'
                    )
                    all_detail_lines.append(
                        f"    Context: ...{entry['context']}..."
                    )
            else:
                all_detail_lines.append(f"\n[{participant_id}] No substitutions made.")

        except Exception as e:
            print(f"ERROR: {e}")
            summary_rows.append({
                "participant_id": participant_id,
                "status": f"error: {e}",
                "participant_turns_total": None,
                "participant_turns_modified": None,
                "total_substitutions": None,
                "substitution_rate": None
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    with open(detail_log_path, "w") as f:
        f.write(f"Variant C Detailed Substitution Log -- {timestamp}\n")
        f.write(f"Rules applied: {len(NORMALIZATION_RULES)}\n")
        f.write("\n".join(all_detail_lines))

    ok_rows = summary_df[summary_df["status"] == "ok"]
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Processed successfully:          {len(ok_rows)}/{len(subject_ids)}")
    if len(ok_rows) > 0:
        n_with_subs = (ok_rows['total_substitutions'] > 0).sum()
        n_zero = (ok_rows['total_substitutions'] == 0).sum()
        total_subs_all = ok_rows['total_substitutions'].sum()
        mean_subs = ok_rows['total_substitutions'].mean()
        print(f"Subjects with >=1 substitution:  {n_with_subs}")
        print(f"Subjects with 0 substitutions:   {n_zero}")
        print(f"Total substitutions (all):       {total_subs_all:.0f}")
        print(f"Mean substitutions per subject:  {mean_subs:.2f}")
    print(f"\nSummary CSV:  {summary_path}")
    print(f"Detail log:   {detail_log_path}")

if __name__ == "__main__":
    main()
