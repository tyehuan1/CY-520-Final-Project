"""
Build the Stage-1 binary dataset: malware (1) vs benign (0).

Combines:
- 7,107 Mal-API-2019 samples (Olivera-preprocessed: first 100 non-repeated
  consecutive API calls) → label "Malware" (1)
- 1,079 Olivera benign samples (already 100 timesteps) → label "Benign" (0)

Both datasets are standardized to the Olivera collection format so that
sequence length cannot serve as a shortcut for the binary classifier.

Preprocessing:
1. Mal-API sequences are Olivera-preprocessed (dedup + first 100 calls).
2. Olivera benign tokens are lowercased to match Mal-API casing.
3. All Mal-API family labels are replaced with "Malware".
4. Stratified 80/20 train/test split.
5. Vocabulary built from training split only.
6. Sequences encoded with the binary vocabulary.

Usage::

    python -m src.data_loading.build_binary_dataset
"""

import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

import config as cfg
from src.data_loading.data_loader import load_mal_api, load_olivera
from src.data_loading.olivera_api_map import OLIVERA_API_DECODE
from src.data_loading.preprocessing import (
    build_vocabulary,
    encode_samples,
    olivera_preprocess_samples,
    save_vocabulary,
    stratified_split,
)
from src.utils import get_logger, save_json, save_pickle

logger = get_logger(__name__)


def build_binary_dataset():
    """Build and cache the binary malware-vs-benign dataset."""

    # ── Load raw data ───────────────────────────────────────────────────
    malapi = load_mal_api()
    olivera = load_olivera(api_decode_map=OLIVERA_API_DECODE)
    olivera_benign = [s for s in olivera if s["label"] == cfg.BENIGN_LABEL]

    logger.info(
        "Raw data: %d Mal-API malware + %d Olivera benign = %d total.",
        len(malapi), len(olivera_benign), len(malapi) + len(olivera_benign),
    )

    # ── Olivera-preprocess Mal-API ──────────────────────────────────────
    # "First 100 non-repeated consecutive API calls" — matches Olivera
    # collection methodology so both classes have comparable lengths.
    malapi_processed = olivera_preprocess_samples(malapi)

    # ── Combine into unified format ─────────────────────────────────────
    combined = []

    # All Mal-API → label "Malware" (sequences already lowercase)
    for s in malapi_processed:
        combined.append({
            "sequence": s["sequence"],
            "label": cfg.MALWARE_LABEL,
        })

    # Olivera benign → lowercase tokens to match Mal-API casing
    for s in olivera_benign:
        combined.append({
            "sequence": [tok.lower() for tok in s["sequence"]],
            "label": cfg.BENIGN_LABEL,
        })

    label_counts = Counter(s["label"] for s in combined)
    logger.info("Combined dataset: %s", dict(label_counts))

    # ── Stratified split ────────────────────────────────────────────────
    train, test = stratified_split(combined)

    train_labels = Counter(s["label"] for s in train)
    test_labels = Counter(s["label"] for s in test)
    logger.info("Train labels: %s", dict(train_labels))
    logger.info("Test labels:  %s", dict(test_labels))

    # ── Label encoder ───────────────────────────────────────────────────
    label_encoder = LabelEncoder()
    label_encoder.fit([cfg.BENIGN_LABEL, cfg.MALWARE_LABEL])
    logger.info("Binary label classes: %s", list(label_encoder.classes_))

    # ── Vocabulary from training split ──────────────────────────────────
    vocab = build_vocabulary(train)

    # ── Encode ──────────────────────────────────────────────────────────
    train = encode_samples(train, vocab)
    test = encode_samples(test, vocab)

    # ── Sequence length analysis ────────────────────────────────────────
    train_malware_lens = [len(s["sequence"]) for s in train
                          if s["label"] == cfg.MALWARE_LABEL]
    train_benign_lens = [len(s["sequence"]) for s in train
                         if s["label"] == cfg.BENIGN_LABEL]

    logger.info(
        "Sequence lengths — Malware: min=%d, median=%d, max=%d | "
        "Benign: min=%d, median=%d, max=%d",
        min(train_malware_lens), int(np.median(train_malware_lens)),
        max(train_malware_lens),
        min(train_benign_lens), int(np.median(train_benign_lens)),
        max(train_benign_lens),
    )

    # ── Cache everything ────────────────────────────────────────────────
    cfg.BINARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    save_pickle(train, cfg.BINARY_PREPROCESSED_TRAIN_PATH)
    save_pickle(test, cfg.BINARY_PREPROCESSED_TEST_PATH)
    save_vocabulary(vocab, cfg.BINARY_VOCABULARY_PATH)
    save_pickle(label_encoder, cfg.BINARY_LABEL_ENCODER_PATH)

    # Save dataset stats
    stats = {
        "total_samples": len(train) + len(test),
        "train_samples": len(train),
        "test_samples": len(test),
        "train_labels": dict(train_labels),
        "test_labels": dict(test_labels),
        "vocab_size": len(vocab),
        "malware_source": "Mal-API-2019 (Olivera-preprocessed: first 100 deduped)",
        "benign_source": "Olivera dataset benign samples",
        "malware_seq_length": {
            "min": min(train_malware_lens),
            "median": int(np.median(train_malware_lens)),
            "mean": round(np.mean(train_malware_lens), 1),
            "max": max(train_malware_lens),
        },
        "benign_seq_length": {
            "min": min(train_benign_lens),
            "median": int(np.median(train_benign_lens)),
            "mean": round(np.mean(train_benign_lens), 1),
            "max": max(train_benign_lens),
        },
    }
    save_json(stats, cfg.BINARY_CACHE_DIR / "dataset_stats.json")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Stage-1 Binary Dataset Summary")
    print(f"{'='*55}")
    print(f"Total:   {len(train) + len(test)} samples")
    print(f"Train:   {len(train)} ({dict(train_labels)})")
    print(f"Test:    {len(test)} ({dict(test_labels)})")
    print(f"Vocab:   {len(vocab)} tokens")
    print(f"\nSequence lengths (train):")
    print(f"  Malware: median={int(np.median(train_malware_lens))}, "
          f"mean={np.mean(train_malware_lens):.0f}, "
          f"max={max(train_malware_lens)}")
    print(f"  Benign:  median={int(np.median(train_benign_lens))}, "
          f"mean={np.mean(train_benign_lens):.0f}, "
          f"max={max(train_benign_lens)}")
    print(f"\nCached to: {cfg.BINARY_CACHE_DIR}")


if __name__ == "__main__":
    build_binary_dataset()
