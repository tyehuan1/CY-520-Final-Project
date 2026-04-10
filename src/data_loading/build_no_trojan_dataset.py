"""
Build Trojan-filtered datasets for Stage-2 (7-class family classification).

Three datasets are processed in a single run, all going through the same
shape: load → drop Trojan → clean/encode against the no-Trojan Mal-API
vocabulary → cache to disk.

* **Mal-API-2019**  — load existing pre-split train/test pickles, drop the
  Trojan rows, rebuild the vocabulary on the filtered training split, and
  re-encode train + test against it.
* **MalbehavD-V1**  — load the labeled JSON, drop Trojan rows, then run the
  same external-dataset preprocessing path used at inference time so the
  cached samples already carry an ``encoded`` field.
* **WinMET (CAPE)** — load directly from the parquet, drop Trojan rows and
  any rows whose primary class isn't one of the 7 Mal-API families, then
  run the same external preprocessing path *with cross-sandbox token
  normalization enabled* and cache the result.

After this script runs, every downstream consumer (training, evaluation,
generalizability) can load a fully-encoded ``.pkl`` / ``.json`` straight
from ``cache/no_trojan/`` without re-running any preprocessing.

Usage::

    python -m src.data_loading.build_no_trojan_dataset
"""

from collections import Counter

from sklearn.preprocessing import LabelEncoder

import config as cfg
from src.data_loading.preprocessing import (
    build_vocabulary,
    encode_samples,
    load_winmet_samples,
    preprocess_external_samples,
)
from src.utils import get_logger, load_json, load_pickle, save_json, save_pickle

logger = get_logger(__name__)

TROJAN_LABEL = "Trojan"


def filter_trojan(samples: list) -> list:
    """Remove samples with label 'Trojan'."""
    return [s for s in samples if s["label"] != TROJAN_LABEL]


def main() -> None:
    """Build no-Trojan filtered datasets for all three sources."""

    # ── Mal-API-2019 ────────────────────────────────────────────────────
    logger.info("Loading existing preprocessed Mal-API data...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)

    train_before = len(train_samples)
    test_before = len(test_samples)

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

    # Rebuild vocabulary from filtered training data so the Trojan-only
    # tokens (if any) drop out and indices stay deterministic.
    logger.info("Rebuilding vocabulary from filtered training data...")
    vocab = build_vocabulary(train_filtered)

    # Strip stale 'encoded' field then re-encode against the new vocab.
    for s in train_filtered:
        s.pop("encoded", None)
    for s in test_filtered:
        s.pop("encoded", None)

    train_encoded = encode_samples(train_filtered, vocab)
    test_encoded = encode_samples(test_filtered, vocab)

    label_encoder = LabelEncoder()
    label_encoder.fit(cfg.MALWARE_FAMILIES)
    logger.info("Label encoder classes: %s", list(label_encoder.classes_))

    train_labels = set(s["label"] for s in train_encoded)
    test_labels = set(s["label"] for s in test_encoded)
    assert train_labels.issubset(set(cfg.MALWARE_FAMILIES)), (
        f"Unexpected train labels: {train_labels - set(cfg.MALWARE_FAMILIES)}"
    )
    assert test_labels.issubset(set(cfg.MALWARE_FAMILIES)), (
        f"Unexpected test labels: {test_labels - set(cfg.MALWARE_FAMILIES)}"
    )

    cfg.NO_TROJAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    save_pickle(train_encoded, cfg.NO_TROJAN_TRAIN_PATH)
    save_pickle(test_encoded, cfg.NO_TROJAN_TEST_PATH)
    save_json(vocab, cfg.NO_TROJAN_VOCABULARY_PATH)
    save_pickle(label_encoder, cfg.NO_TROJAN_LABEL_ENCODER_PATH)

    logger.info("No-Trojan Mal-API data saved to %s", cfg.NO_TROJAN_CACHE_DIR)

    # ── MalbehavD-V1 ────────────────────────────────────────────────────
    logger.info("Loading MalBehavD labeled data...")
    malbehavd_data = load_json(cfg.MALBEHAVD_LABELED_PATH)
    all_mb_samples = malbehavd_data["samples"]

    mb_before = len([s for s in all_mb_samples if s["label"] != cfg.BENIGN_LABEL])
    mb_filtered = [s for s in all_mb_samples if s["label"] != TROJAN_LABEL]
    mb_malware_after = len(
        [s for s in mb_filtered if s["label"] != cfg.BENIGN_LABEL]
    )
    mb_trojan = mb_before - mb_malware_after

    logger.info(
        "MalBehavD malware: %d -> %d (removed %d Trojan). Benign preserved.",
        mb_before, mb_malware_after, mb_trojan,
    )

    # Same preprocessing path as inference.  MalBehavD is Cuckoo-sourced so
    # the API names already match the Mal-API vocabulary; cross-sandbox
    # normalization is not needed here (and would be a no-op).
    mb_encoded = preprocess_external_samples(
        mb_filtered, vocab,
        normalize_for_vocab=False,
        dataset_name="MalBehavD-V1",
    )

    mb_label_counts = Counter(s["label"] for s in mb_encoded)
    filtered_malbehavd = {
        "samples": mb_encoded,
        "dropped_hashes": malbehavd_data.get("dropped_hashes", []),
        "stats": {
            "total_labeled": len(mb_encoded),
            "total_dropped": len(malbehavd_data.get("dropped_hashes", [])),
            "family_distribution": dict(mb_label_counts.most_common()),
            "trojan_removed": mb_trojan,
        },
    }
    save_json(filtered_malbehavd, cfg.NO_TROJAN_MALBEHAVD_PATH)
    logger.info(
        "No-Trojan MalBehavD data saved to %s", cfg.NO_TROJAN_MALBEHAVD_PATH,
    )

    # ── WinMET (CAPE) ───────────────────────────────────────────────────
    # Load directly from the parquet so we keep the secondary_classes /
    # family_avclass / sha256 metadata that the misclassification analysis
    # in evaluate_generalizability.py needs.
    logger.info("Loading WinMET samples (drop_trojan=True)...")
    wm_samples_raw = load_winmet_samples(drop_trojan=True)
    wm_before = len(wm_samples_raw)

    # Drop any sample whose label isn't one of the model's 7 classes.  This
    # mirrors the guard previously kept inline in evaluate_generalizability.py
    # but does it once at build time so the cached file is already clean.
    known = set(cfg.MALWARE_FAMILIES)
    wm_samples_raw = [s for s in wm_samples_raw if s["label"] in known]
    logger.info(
        "WinMET kept %d/%d after class filter (model classes only).",
        len(wm_samples_raw), wm_before,
    )

    # Cross-sandbox normalization ON for WinMET — CAPE emits Win32-layer
    # API names that need to be remapped onto the Cuckoo NT-layer vocab.
    wm_encoded = preprocess_external_samples(
        wm_samples_raw, vocab,
        normalize_for_vocab=True,
        dataset_name="WinMET",
    )

    save_pickle(wm_encoded, cfg.NO_TROJAN_WINMET_PATH)
    logger.info(
        "No-Trojan WinMET data saved to %s", cfg.NO_TROJAN_WINMET_PATH,
    )

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
    print(f"  Benign:  {len([s for s in mb_encoded if s['label'] == cfg.BENIGN_LABEL])}")
    print(f"  Distribution:")
    for fam, count in mb_label_counts.most_common():
        print(f"    {fam:>12}: {count}")

    print(f"\nWinMET (cross-dataset generalizability):")
    print(f"  Samples: {len(wm_encoded)}")
    wm_dist = Counter(s["label"] for s in wm_encoded)
    print(f"  Distribution:")
    for fam, count in wm_dist.most_common():
        print(f"    {fam:>12}: {count}")


if __name__ == "__main__":
    main()
