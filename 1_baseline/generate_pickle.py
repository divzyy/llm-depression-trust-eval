"""
generate_pickle.py  (v2 — corrected split)

Standalone script version of the pickle generation from the baseline notebook
(embedding_quantitative_analysis.ipynb) by Greene & Blair (2025).

CHANGE LOG vs v1
----------------
v1 reproduced the notebook's runtime stratified split (STEP 2 of the original
script). Running that split verbatim does NOT reproduce the split the authors
actually used: its test set matches the authors' released test set on only 39
of 41 subjects, and assigns participants 339 and 345 (members of the authors'
actual test set) to the training side. As a result, the v1 pickle contained
test subjects 339 and 345 in the few-shot knowledge base (verified 13-06-2026).

v2 therefore:
  1. REPLACES the runtime split with the authors' actual split, reconstructed
     from the artifacts in their released repository (test IDs from their
     released test-set result files; validation IDs from the union of their
     released VAL_analysis_output run files; training = the remaining 58 of
     the 142 train+dev subjects). The split lives in split_manifest.py — the
     single source of truth. No split logic remains in this script.
  2. REFUSES to resume from an existing pickle that contains any non-training
     participant (so an old contaminated pickle cannot silently survive via
     the resume mechanism).
  3. VERIFIES the final pickle: exactly the 58 training participants, zero
     overlap with the test set, and fails loudly otherwise.

All embedding/chunking functions below are copied exactly from the notebook.
Only paths/config were adapted for the Alice cluster (as in v1).

Produces: chunk_8_step_2_participant_embedded_transcripts.pkl
containing the 58 training participants (the knowledge base for few-shot
retrieval).

NOTE BEFORE RUNNING: move the old contaminated pickle out of the way first,
e.g.
  mv agents/chunk_8_step_2_participant_embedded_transcripts.pkl \
     agents/chunk_8_step_2_participant_embedded_transcripts_LEAKY_v1.pkl
(keep it — it is evidence for the thesis). The script will refuse to run
against it anyway, but renaming keeps things tidy.
"""

import json
import requests
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import pickle
import os
import sys
import math

ROOT = os.environ.get("AIPSY_ROOT", os.path.expanduser("~/ai-psychiatrist"))
DAIC = os.environ.get("DAIC_ROOT", os.path.expanduser("~/daic_woz_data"))

# CONFIG — adapted for the Alice cluster

OLLAMA_NODE = os.environ.get("OLLAMA_NODE", "localhost")

DEV_CSV   = f"{DAIC}/labels/dev_split_Depression_AVEC2017.csv"
TRAIN_CSV = f"{DAIC}/labels/train_split_Depression_AVEC2017.csv"
TRANSCRIPT_DIR = f"{DAIC}/transcripts"

# Set to True if transcripts are in subdirectories like {id}_P/{id}_TRANSCRIPT.csv
# Set to False if transcripts are flat like transcripts/{id}_TRANSCRIPT.csv
TRANSCRIPTS_IN_SUBDIRS = False

pickle_file = f"{ROOT}/agents/chunk_8_step_2_participant_embedded_transcripts.pkl"
dim = 4096

# STEP 1: Load PHQ-8 ground truths (exact notebook code)

dev_split_phq8 = pd.read_csv(DEV_CSV)
train_split_phq8 = pd.read_csv(TRAIN_CSV)
phq8_ground_truths = pd.concat([dev_split_phq8, train_split_phq8], ignore_index=True)
phq8_ground_truths = phq8_ground_truths.sort_values('Participant_ID').reset_index(drop=True)

print(f"Total subjects with PHQ-8 data: {len(phq8_ground_truths)}")

# STEP 2 (v2): Load the authors' actual split from the manifest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from split_manifest import TRAIN_IDS, VAL_IDS, TEST_IDS, check_integrity
except ImportError:
    sys.exit(
        "ERROR: split_manifest.py not found next to generate_pickle.py.\n"
        "It contains the reconstructed authors' split (58/43/41) and is the\n"
        "single source of truth for which subjects enter the knowledge base."
    )

check_integrity(set(phq8_ground_truths['Participant_ID'].astype(int)))

train_ids = list(TRAIN_IDS)
val_ids = list(VAL_IDS)
test_ids = list(TEST_IDS)

assert len(train_ids) == 58, "training set must contain exactly 58 participants"
assert not set(train_ids) & set(test_ids), "training set overlaps test set!"
assert 339 in test_ids and 345 in test_ids, "339/345 must be test subjects"
assert 339 not in train_ids and 345 not in train_ids, "339/345 leaked into train!"

print(f"\nSplit loaded from manifest:")
print(f"Train: {len(train_ids)}  Validation: {len(val_ids)}  Test: {len(test_ids)}")

split_manifest_file = f"{ROOT}/agents/split_manifest_58_43_41.json"

with open(split_manifest_file, "w") as f:
    json.dump(
        {
            "provenance": "authors' actual split reconstructed from released "
                          "artifacts (test: released test result files; val: "
                          "union of released VAL_analysis_output files; train: "
                          "remaining 58 of 142). See split_manifest.py.",
            "train_ids": sorted(train_ids),
            "val_ids": sorted(val_ids),
            "test_ids": sorted(test_ids)
        },
        f,
        indent=2
    )

print(f"Saved split manifest to: {split_manifest_file}")

# STEP 3: Grab knowledgebase transcripts (exact notebook code, paths adapted)

participant_transcripts = {}

for participant_id in train_ids:
    try:
        if TRANSCRIPTS_IN_SUBDIRS:
            transcript_path = os.path.join(TRANSCRIPT_DIR, f"{participant_id}_P", f"{participant_id}_TRANSCRIPT.csv")
        else:
            transcript_path = os.path.join(TRANSCRIPT_DIR, f"{participant_id}_TRANSCRIPT.csv")

        current_transcript = pd.read_csv(transcript_path, sep="\t")

        current_transcript['speaker'] = current_transcript['speaker'].fillna('Unknown').astype(str)
        current_transcript['value'] = current_transcript['value'].fillna('').astype(str)

        current_patient_transcript = '\n'.join(current_transcript['speaker'] + ': ' + current_transcript['value'])
        participant_transcripts[participant_id] = current_patient_transcript

    except FileNotFoundError:
        print(f"File for participant {participant_id} not found")
        participant_transcripts[participant_id] = None
    except Exception as e:
        print(f"Error processing participant {participant_id}: {e}")
        participant_transcripts[participant_id] = None

missing = sorted(pid for pid, t in participant_transcripts.items() if t is None)
if missing:
    sys.exit(f"ERROR: missing/unreadable transcripts for training subjects: {missing}")

print(len(participant_transcripts))
print(f"{'-'*30}\n{len(train_ids)}")
print(sorted(train_ids))

# STEP 4: Embed knowledgebase transcripts (exact notebook functions)

def load_existing_embeddings(pickle_file):
    """
    Loads embedded reference transcripts from pickle file

    Parameters
    ----------
    pickle_file : string
        path to the pickle file

    Returns
    -------
    dict
        Dict of embedded chunks of the reference transcripts and their participant IDs
    """
    if os.path.exists(pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
        except:
            print(f"Error loading {pickle_file}")
            return {}
    return {}

def save_embeddings(embeddings_dict, pickle_file):
    """
    Saves an embedding to the pickle file

    Parameters
    ----------
    embeddings_dict : dict
        The dictionary with participant IDs as keys and (raw_text, embedding) as values for each transcript chunk

    Writes
    -------
    dict
        Adds the embeddings to the pickle file
    """
    with open(pickle_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)

def get_embedding(text, model="qwen3-embedding:8b-q8_0", dim=None):
    """
    Creates embedding from given text input and model 

    Parameters
    ----------
    text : string
        The text to be embedded
    model : string
        The name of the ollama model to be used for embedding
    dim : int, optional
        If provided, truncate to this dimension and normalize (MRL support)

    Returns
    -------
    list
        The vector embedding of the text
    """
    BASE_URL = f"http://{OLLAMA_NODE}:11434/api/embeddings"

    response = requests.post(
        BASE_URL,
        json={
            "model": model,
            "prompt": text
        }
    )

    if response.status_code == 200:
        embedding = response.json()["embedding"]

        # Manually setting dimension because ollama doesn't natively support atm
        if dim is not None:
            # Truncate and normalize for MRL models
            embedding = embedding[:dim]
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

        return embedding
    else:
        raise Exception(f"API call failed with status {response.status_code}: {response.text}")

def create_sliding_chunks(transcript_text, chunk_size=8, step_size=2):
    """
    Splits the transcript into several chunks 

    Parameters
    ----------
    transcript_text : string
        The transcript
    chunk_size : int
        The amount of newlines per chunk
    step_size : int
        The newline distance moved each time a chunk is created
            -Ex. transcript_text = "A\nB\nC\nD\nE\nF\nG\nH", chunk_size = 4, step_size = 2
            -Chunk 1: "A\nB\nC\nD"
            -Chunk 2: "C\nD\nE\nF"

    Returns
    -------
    list
        The text chunk strings
    """
    lines = transcript_text.split('\n')

    while lines and lines[-1] == '':
        lines.pop()

    chunks = []

    if len(lines) <= chunk_size:
        return ['\n'.join(lines)]

    for i in range(0, len(lines) - chunk_size + 1, step_size):
        chunk = '\n'.join(lines[i:i + chunk_size])
        chunks.append(chunk)

    last_chunk_start = len(lines) - chunk_size
    if last_chunk_start > 0 and (last_chunk_start % step_size) != 0:
        final_chunk = '\n'.join(lines[last_chunk_start:])
        if final_chunk not in chunks:
            chunks.append(final_chunk)

    return chunks

def process_transcripts(participant_transcripts, pickle_file, dim):
    """
    Chunking and embedding the reference transcripts and saving them to the pickle file

    Parameters
    ----------
    participant_transcripts : dict
        A dict of participant transcripts with the key being participant id and value being the transcript
    pickle_file : str
        The path to the pickle file

    Returns
    -------
    dict
        The dictionary with participant IDs as keys and (raw_text, embedding) as values for each transcript chunk
    """
    participant_embedded_transcripts = load_existing_embeddings(pickle_file)

    if participant_embedded_transcripts:
        existing = set(int(k) for k in participant_embedded_transcripts.keys())
        intruders = sorted(existing - set(int(i) for i in TRAIN_IDS))
        if intruders:
            sys.exit(
                f"ERROR: existing pickle at {pickle_file} contains non-training "
                f"subjects {intruders}.\nMove it out of the way first, e.g.:\n"
                f"  mv {pickle_file} {pickle_file.replace('.pkl', '_LEAKY_v1.pkl')}"
            )
        print(f"Resuming from existing clean pickle with {len(existing)} subjects.")

    for participant_id, transcript in participant_transcripts.items():
        if participant_id in participant_embedded_transcripts:
            print(f"Skipping participant {participant_id} - already processed")
            continue

        print(f"Processing participant {participant_id}...")

        try:
            chunks = create_sliding_chunks(transcript)

            embeddings_list = []
            for i, chunk in enumerate(chunks):
                print(f"  Processing chunk {i+1}/{len(chunks)}")
                embedding = get_embedding(chunk,dim=dim)
                embeddings_list.append((chunk, embedding))

            participant_embedded_transcripts[participant_id] = np.array(embeddings_list, dtype=object)

            save_embeddings(participant_embedded_transcripts, pickle_file)
            print(f"Completed participant {participant_id} - saved to {pickle_file}")

        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")
            print("Stopping processing and saving current progress...")
            save_embeddings(participant_embedded_transcripts, pickle_file)
            break

    return participant_embedded_transcripts

os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
participant_embedded_transcripts = process_transcripts(participant_transcripts, pickle_file, dim)

final_ids = set(int(k) for k in participant_embedded_transcripts.keys())
overlap = sorted(final_ids & set(int(i) for i in TEST_IDS))

print("\n" + "=" * 60)
print("FINAL PICKLE VERIFICATION")
print(f"  subjects in pickle : {len(final_ids)}")
print(f"  overlap with test  : {overlap}")
print("=" * 60)

assert len(final_ids) == 58, f"pickle has {len(final_ids)} subjects, expected 58 (incomplete run?)"
assert not overlap, f"LEAKAGE: test subjects {overlap} in knowledge base!"
assert final_ids == set(int(i) for i in TRAIN_IDS), "pickle subjects != manifest training set"
print("OK: knowledge base contains exactly the 58 training participants, no test overlap.")
