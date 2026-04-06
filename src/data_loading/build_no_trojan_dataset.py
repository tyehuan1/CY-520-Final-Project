"""
Build Trojan-filtered datasets for Stage-2 (7-class family classification).

Reads the existing preprocessed Mal-API train/test data and MalBehavD labeled
data, removes all samples labeled "Trojan", rebuilds the vocabulary and label
encoder, and re-encodes sequences.

This preserves the original train/test split — we only remove Trojan samples,
we don't re-split.

Usage::

    python -m src.data_loading.build_no_trojan_dataset
"""

from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder

import config as cfg
from src.data_loading.preprocessing import build_vocabulary, encode_samples
from src.utils import get_logger, load_json, load_pickle, save_json, save_pickle

logger = get_logger(__name__)

TROJAN_LABEL = "Trojan"


def filter_trojan(samples: list) -> list:
    """Remove samples with label 'Trojan'."""
    filtered = [s for s in samples if s["label"] != TROJAN_LABEL]
    return filtered


def main() -> None:
    """Build no-Trojan filtered datasets."""

    # ── Load existing preprocessed Mal-API data ─────────────────────────
    logger.info("Loading existing preprocessed Mal-API data...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)

    train_before = len(train_samples)
    test_before = len(test_samples)

    # Filter out Trojan
    train_filtered = filter_trojan(train_samples)
    test_filtered = filter_trojan(test_samples)

    trojan_train = train_before - len(train_filtered)
    trojan_test = test_before - len(test_filtered)

    logger.info(
        "Mal-API train: %d -> %d (removed %d Trojan).",
        train_before, len(train_filtered), trojan_train,
    )
    logger.info(
        "Mal-API test: %d -> %d (removed %d Trojan).",
        test_before, len(test_filtered), trojan_test,
    )

    # ── Rebuild vocabulary from filtered training data ───────────────────
    # The vocabulary doesn't change much (Trojan samples use similar API
    # calls), but rebuilding ensures consistency.
    logger.info("Rebuilding vocabulary from filtered training data...")
    vocab = build_vocabulary(train_filtered)

    # ── Re-encode sequences with the new vocabulary ─────────────────────
    # Strip existing 'encoded' field and re-encode
    for s in train_filtered:
        s.pop("encoded", None)
    for s in test_filtered:
        s.pop("encoded", None)

    train_encoded = encode_samples(train_filtered, vocab)
    test_encoded = encode_samples(test_filtered, vocab)

    # ── Build label encoder ─────────────────────────────────────────────
    label_encoder = LabelEncoder()
    label_encoder.fit(cfg.MALWARE_FAMILIES)
    logger.info("Label encoder classes: %s", list(label_encoder.classes_))

    # Verify all labels in the data match the 7 families
    train_labels = set(s["label"] for s in train_encoded)
    test_labels = set(s["label"] for s in test_encoded)
    assert train_labels.issubset(set(cfg.MALWARE_FAMILIES)), (
        f"Unexpected train labels: {train_labels - set(cfg.MALWARE_FAMILIES)}"
    )
    assert test_labels.issubset(set(cfg.MALWARE_FAMILIES)), (
        f"Unexpected test labels: {test_labels - set(cfg.MALWARE_FAMILIES)}"
    )

    # ── Save filtered data ──────────────────────────────────────────────
    cfg.NO_TROJAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    save_pickle(train_encoded, cfg.NO_TROJAN_TRAIN_PATH)
    save_pickle(test_encoded, cfg.NO_TROJAN_TEST_PATH)
    save_json(vocab, cfg.NO_TROJAN_VOCABULARY_PATH)
    save_pickle(label_encoder, cfg.NO_TROJAN_LABEL_ENCODER_PATH)

    logger.info("No-Trojan Mal-API data saved to %s", cfg.NO_TROJAN_CACHE_DIR)

    # ── Filter MalBehavD labeled data ───────────────────────────────────
    logger.info("Loading MalBehavD labeled data...")
    malbehavd_data = load_json(cfg.MALBEHAVD_LABELED_PATH)
    all_samples = malbehavd_data["samples"]

    mb_before = len([s for s in all_samples if s["label"] != cfg.BENIGN_LABEL])
    filtered_samples = [
        s for s in all_samples if s["label"] != TROJAN_LABEL
    ]
    mb_malware_after = len([s for s in filtered_samples if s["label"] != cfg.BENIGN_LABEL])
    mb_trojan = mb_before - mb_malware_after

    logger.info(
        "MalBehavD malware: %d -> %d (removed %d Trojan). Benign preserved.",
        mb_before, mb_malware_after, mb_trojan,
    )

    # Rebuild stats
    label_counts = Counter(s["label"] for s in filtered_samples)
    filtered_malbehavd = {
        "samples": filtered_samples,
        "dropped_hashes": malbehavd_data.get("dropped_hashes", []),
        "stats": {
            "total_labeled": len(filtered_samples),
            "total_dropped": len(malbehavd_data.get("dropped_hashes", [])),
            "family_distribution": dict(label_counts.most_common()),
            "trojan_removed": mb_trojan,
        },
    }

    save_json(filtered_malbehavd, cfg.NO_TROJAN_MALBEHAVD_PATH)
    logger.info("No-Trojan MalBehavD data saved to %s", cfg.NO_TROJAN_MALBEHAVD_PATH)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("No-Trojan Dataset Summary")
    print(f"{'='*55}")
    print(f"\nMal-API (Stage-2 training/test):")
    print(f"  Train: {train_before} -> {len(train_encoded)} ({trojan_train} Trojan removed)")
    print(f"  Test:  {test_before} -> {len(test_encoded)} ({trojan_test} Trojan removed)")
    print(f"  Vocab: {len(vocab)} tokens")
    print(f"  Classes: {list(label_encoder.classes_)}")

    train_dist = Counter(s["label"] for s in train_encoded)
    test_dist = Counter(s["label"] for s in test_encoded)
    print(f"\n  Train distribution:")
    for fam in sorted(train_dist):
        print(f"    {fam:>12}: {train_dist[fam]}")
    print(f"\n  Test distribution:")
    for fam in sorted(test_dist):
        print(f"    {fam:>12}: {test_dist[fam]}")

    print(f"\nMalBehavD (generalizability):")
    print(f"  Malware: {mb_before} -> {mb_malware_after} ({mb_trojan} Trojan removed)")
    print(f"  Benign:  {len([s for s in filtered_samples if s['label'] == cfg.BENIGN_LABEL])}")
    print(f"  Distribution:")
    for fam, count in label_counts.most_common():
        print(f"    {fam:>12}: {count}")


if __name__ == "__main__":
    main()
