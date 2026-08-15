"""
split_manifest.py — the authors' actual train/val/test split of the 142 DAIC-WOZ
subjects, reconstructed from the artifacts released in the original repository
(github.com/trendscenter/ai-psychiatrist):

  TEST_IDS  : the 41 participant IDs present in the authors' released test-set
              result files (metareview_gemma_few_shot.csv, TEST/DIM_TEST jsonls,
              quan_medgemma_*.jsonl).
  VAL_IDS   : the union of participant IDs in the authors' released
              VAL_analysis_output run files (exactly 43, disjoint from TEST_IDS).
  TRAIN_IDS : the remaining 58 of the 142 train+dev subjects.

This file is the single source of truth for the split. The few-shot retrieval
knowledge base (pickle) must be built from TRAIN_IDS only.

Verified on reconstruction: len(TRAIN)=58, len(VAL)=43, len(TEST)=41,
pairwise disjoint, union = all 142 subjects, and 339/345 are in TEST (not TRAIN).
"""

TRAIN_IDS = [303, 304, 305, 310, 312, 313, 315, 317, 318, 321, 324, 327, 335, 338, 340, 343, 344, 346, 347, 350, 352, 356, 363, 368, 369, 388, 391, 395, 397, 400, 402, 404, 406, 412, 414, 415, 416, 418, 426, 429, 433, 434, 437, 439, 444, 458, 463, 464, 473, 474, 475, 476, 477, 478, 483, 486, 488, 491]

VAL_IDS = [302, 307, 320, 322, 325, 326, 328, 331, 333, 336, 341, 348, 351, 353, 355, 358, 360, 364, 366, 371, 372, 374, 376, 380, 381, 382, 392, 401, 403, 419, 420, 425, 440, 443, 446, 448, 454, 457, 471, 479, 482, 490, 492]

TEST_IDS = [316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383, 385, 386, 389, 390, 393, 409, 413, 417, 422, 423, 427, 428, 430, 436, 441, 445, 447, 449, 451, 455, 456, 459, 468, 472, 484, 485, 487, 489]

def check_integrity(all_subject_ids=None):
    """Assert the manifest is internally consistent. Optionally pass the full
    set of 142 subject IDs (from the AVEC label files) to verify coverage."""
    t, v, te = set(TRAIN_IDS), set(VAL_IDS), set(TEST_IDS)
    assert len(TRAIN_IDS) == 58 and len(VAL_IDS) == 43 and len(TEST_IDS) == 41
    assert not (t & v) and not (t & te) and not (v & te), "splits overlap!"
    if all_subject_ids is not None:
        assert t | v | te == set(all_subject_ids), "manifest does not cover all subjects"
    return True

if __name__ == "__main__":
    check_integrity()
    print("split manifest OK: 58 train / 43 val / 41 test, pairwise disjoint")
