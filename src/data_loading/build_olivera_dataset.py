"""
Build the aligned Olivera generalizability dataset.

Produces three artifacts:
  1. **Training data** — Mal-API samples preprocessed to match Olivera format
     (consecutive-dedup, first 100 calls, lowercased), split 80/20.
  2. **External test data** — VT-labeled Olivera malware (lowercased, artifacts
     removed), filtered to families present in the training vocabulary.
  3. **Shared vocabulary** — built from the Mal-API training split only, so the
     Olivera test set is truly unseen.

Both datasets end up in the same token space: lowercase Cuckoo API names,
max 100 calls, no sandbox artifacts, no consecutive duplicates.

Usage::

    python -m src.data_loading.build_olivera_dataset
"""

from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config as cfg
from src.data_loading.preprocessing import (
    olivera_preprocess_samples,
    remove_sandbox_tokens,
)
from src.utils import get_logger, load_json, load_pickle, save_json, save_pickle

logger = get_logger(__name__)

Sample = Dict[str, object]

# Minimum samples per class in Olivera to include in evaluation.
# Classes below this threshold are too small for meaningful metrics.
MIN_OLIVERA_CLASS_SIZE = 10


def _lowercase_sequences(samples: List[Sample]) -> List[Sample]:
    """Lowercase all tokens in each sample's sequence.

    Args:
        samples: Samples with ``sequence`` field.

    Returns:
        New list of samples with lowercased sequences.
    """
    out = []
    for s in samples:
        new = dict(s)
        new["sequence"] = [t.lower() for t in s["sequence"]]
        out.append(new)
    return out


def _clean_olivera_sequences(samples: List[Sample]) -> List[Sample]:
    """Clean Olivera test sequences: remove sandbox artifacts, lowercase.

    Args:
        samples: Raw Olivera samples with PascalCase tokens.

    Returns:
        Cleaned samples with lowercase tokens, artifacts removed.
    """
    out = []
    for s in samples:
        seq = remove_sandbox_tokens(s["sequence"])
        new = dict(s)
        new["sequence"] = [t.lower() for t in seq]
        out.append(new)
    return out


def _build_vocabulary(samples: List[Sample]) -> Dict[str, int]:
    """Build token→index vocabulary from training samples.

    Reserves index 0 for <PAD> and 1 for <UNK>.

    Args:
        samples: Training samples with ``sequence`` field.

    Returns:
        Vocabulary mapping token string to integer index.
    """
    token_counts: Counter = Counter()
    for s in samples:
        token_counts.update(s["sequence"])

    vocab = {cfg.PAD_TOKEN: cfg.PAD_INDEX, cfg.UNK_TOKEN: cfg.UNK_INDEX}
    for idx, (token, _) in enumerate(token_counts.most_common(), start=2):
        vocab[token] = idx

    return vocab


def _encode_samples(
    samples: List[Sample],
    vocab: Dict[str, int],
) -> List[Sample]:
    """Integer-encode sample sequences using the given vocabulary.

    Args:
        samples: Samples with ``sequence`` field.
        vocab: Token→index mapping.

    Returns:
        Samples with added ``encoded`` field (list of ints).
    """
    unk_idx = vocab[cfg.UNK_TOKEN]
    out = []
    for s in samples:
        new = dict(s)
        new["encoded"] = [vocab.get(t, unk_idx) for t in s["sequence"]]
        out.append(new)
    return out


def main() -> None:
    cfg.OLIVERA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 1. Prepare Mal-API training data (Olivera-format)
    # ==================================================================
    logger.info("Loading Olivera-limited Mal-API data...")
    malapi_samples = load_pickle(cfg.MALAPI_OLIVERA_LIMITED_PATH)
    logger.info("Loaded %d Mal-API samples.", len(malapi_samples))

    # Lowercase (Mal-API tokens are already lowercase, but be explicit)
    malapi_samples = _lowercase_sequences(malapi_samples)

    # Drop empty sequences
    before = len(malapi_samples)
    malapi_samples = [s for s in malapi_samples if len(s["sequence"]) > 0]
    if len(malapi_samples) < before:
        logger.info(
            "Dropped %d empty sequences.", before - len(malapi_samples),
        )

    # Train/test split (stratified by label)
    labels = [s["label"] for s in malapi_samples]
    train_samples, test_samples = train_test_split(
        malapi_samples,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_SEED,
        stratify=labels,
    )
    logger.info(
        "Mal-API split: %d train, %d test.", len(train_samples), len(test_samples),
    )

    # Build vocabulary from training split only
    vocab = _build_vocabulary(train_samples)
    logger.info("Vocabulary: %d tokens (incl. PAD, UNK).", len(vocab))

    # Encode
    train_samples = _encode_samples(train_samples, vocab)
    test_samples = _encode_samples(test_samples, vocab)

    # Label encoder (8 classes, with Trojan)
    all_labels = sorted(set(s["label"] for s in train_samples))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    logger.info("Classes: %s", list(label_encoder.classes_))

    # Save training artifacts
    save_pickle(train_samples, cfg.OLIVERA_TRAIN_PATH)
    save_pickle(test_samples, cfg.OLIVERA_TEST_PATH)
    save_json(vocab, cfg.OLIVERA_VOCABULARY_PATH)
    save_pickle(label_encoder, cfg.OLIVERA_LABEL_ENCODER_PATH)

    # ==================================================================
    # 2. Prepare Olivera external test data
    # ==================================================================
    logger.info("Loading VT-labeled Olivera data...")
    olivera_data = load_json(cfg.OLIVERA_VT_LABELED_PATH)
    olivera_all = olivera_data["samples"]

    # Keep malware only
    olivera_malware = [s for s in olivera_all if s["label"] != cfg.BENIGN_LABEL]
    logger.info("Olivera malware samples: %d", len(olivera_malware))

    # Clean: remove __exception__/__anomaly__, lowercase
    olivera_malware = _clean_olivera_sequences(olivera_malware)

    # Filter to classes present in training data
    known_classes = set(label_encoder.classes_)
    olivera_malware = [s for s in olivera_malware if s["label"] in known_classes]
    logger.info(
        "After class filter (%s): %d samples.",
        known_classes, len(olivera_malware),
    )

    # Filter out classes with too few samples for meaningful evaluation
    class_counts = Counter(s["label"] for s in olivera_malware)
    small_classes = {
        c for c, n in class_counts.items() if n < MIN_OLIVERA_CLASS_SIZE
    }
    if small_classes:
        logger.warning(
            "Excluding classes with < %d samples: %s",
            MIN_OLIVERA_CLASS_SIZE, small_classes,
        )
        olivera_malware = [
            s for s in olivera_malware if s["label"] not in small_classes
        ]

    # Encode with the Mal-API-derived vocabulary
    olivera_malware = _encode_samples(olivera_malware, vocab)

    # Compute UNK statistics
    unk_idx = vocab[cfg.UNK_TOKEN]
    total_tokens = sum(len(s["encoded"]) for s in olivera_malware)
    unk_tokens = sum(
        sum(1 for t in s["encoded"] if t == unk_idx) for s in olivera_malware
    )
    unk_pct = 100 * unk_tokens / total_tokens if total_tokens > 0 else 0
    logger.info(
        "Olivera UNK rate: %d / %d tokens (%.2f%%).",
        unk_tokens, total_tokens, unk_pct,
    )

    save_pickle(olivera_malware, cfg.OLIVERA_EXT_TEST_PATH)

    # ==================================================================
    # Summary
    # ==================================================================
    olivera_dist = Counter(s["label"] for s in olivera_malware)
    train_dist = Counter(s["label"] for s in train_samples)

    print(f"\n{'='*60}")
    print("Olivera Experiment — Dataset Build")
    print(f"{'='*60}")
    print(f"Vocabulary: {len(vocab)} tokens")
    print(f"Mal-API train: {len(train_samples)}  |  Mal-API test: {len(test_samples)}")
    print(f"Olivera external test: {len(olivera_malware)}")
    print(f"Olivera UNK rate: {unk_pct:.2f}%")
    if small_classes:
        print(f"Excluded small classes: {small_classes}")
    print(f"\n{'Class':>12}  {'Train':>6}  {'MalAPI Test':>11}  {'Olivera':>8}")
    print(f"{'-'*42}")
    for cls in label_encoder.classes_:
        t = train_dist.get(cls, 0)
        m = Counter(s["label"] for s in test_samples).get(cls, 0)
        o = olivera_dist.get(cls, 0)
        print(f"{cls:>12}  {t:>6}  {m:>11}  {o:>8}")

    print(f"\nArtifacts saved to: {cfg.OLIVERA_CACHE_DIR}")


if __name__ == "__main__":
    main()
