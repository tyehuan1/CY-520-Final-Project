"""
Preprocessing pipeline: cleaning, vocabulary building, encoding, splitting.

All transformations are deterministic given the random seed in config.
The vocabulary is built exclusively from the training split.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import config as cfg
from src.utils import get_logger, load_json, load_pickle, save_json, save_pickle

logger = get_logger(__name__)

Sample = Dict[str, Any]


# ── Cleaning ───────────────────────────────────────────────────────────────


def remove_sandbox_tokens(sequence: List[str]) -> List[str]:
    """Remove sandbox artifact tokens (anything starting with ``__``).

    Args:
        sequence: Raw API call sequence.

    Returns:
        Filtered sequence with sandbox tokens removed.
    """
    return [tok for tok in sequence if not tok.startswith("__")]


def collapse_consecutive_duplicates(
    sequence: List[str],
    max_repeats: int = cfg.MAX_CONSECUTIVE_DUPLICATES,
) -> List[str]:
    """Collapse runs of the same API call to at most *max_repeats*.

    Args:
        sequence: API call sequence.
        max_repeats: Maximum allowed consecutive repeats of any single call.

    Returns:
        Sequence with long runs truncated.
    """
    if not sequence:
        return []

    result: List[str] = []
    run_count = 0
    prev = None

    for tok in sequence:
        if tok == prev:
            run_count += 1
            if run_count <= max_repeats:
                result.append(tok)
        else:
            result.append(tok)
            prev = tok
            run_count = 1

    return result


def clean_sequence(
    sequence: List[str],
    max_repeats: int = cfg.MAX_CONSECUTIVE_DUPLICATES,
) -> List[str]:
    """Apply all cleaning steps to a single sequence.

    Steps (in order):
    1. Remove sandbox artifact tokens (``__exception__``, ``__anomaly__``, etc.)
    2. Collapse consecutive duplicate API calls to at most *max_repeats*.

    Args:
        sequence: Raw API call sequence.
        max_repeats: Maximum consecutive repeats allowed.

    Returns:
        Cleaned sequence.
    """
    seq = remove_sandbox_tokens(sequence)
    seq = collapse_consecutive_duplicates(seq, max_repeats)
    return seq


def clean_samples(
    samples: List[Sample],
    max_repeats: int = cfg.MAX_CONSECUTIVE_DUPLICATES,
) -> List[Sample]:
    """Clean all samples in-place-safe (returns new list of dicts).

    Args:
        samples: List of sample dicts with ``sequence`` key.
        max_repeats: Maximum consecutive repeats allowed.

    Returns:
        New list of sample dicts with cleaned sequences.
    """
    cleaned = []
    for s in samples:
        new_sample = dict(s)
        new_sample["sequence"] = clean_sequence(s["sequence"], max_repeats)
        cleaned.append(new_sample)
    logger.info("Cleaned %d samples.", len(cleaned))
    return cleaned


# ── Olivera-style preprocessing ──────────────────────────────────────────


def deduplicate_consecutive(sequence: List[str]) -> List[str]:
    """Remove all consecutive duplicate API calls (keep first occurrence).

    Unlike ``collapse_consecutive_duplicates`` which allows up to N repeats,
    this removes ALL consecutive repetitions — matching the Olivera dataset
    description: "first 100 **non-repeated** consecutive API calls."

    Args:
        sequence: API call sequence.

    Returns:
        Sequence with no consecutive duplicates.
    """
    if not sequence:
        return []
    result = [sequence[0]]
    for tok in sequence[1:]:
        if tok != result[-1]:
            result.append(tok)
    return result


def olivera_style_preprocess(
    sequence: List[str],
    max_calls: int = cfg.OLIVERA_SEQ_COLUMNS,
) -> List[str]:
    """Preprocess a sequence to match the Olivera dataset format.

    Applies the Olivera collection methodology: "Each API call sequence is
    composed of the first 100 non-repeated consecutive API calls associated
    with the parent process."

    Steps:
    1. Remove sandbox artifact tokens (``__exception__``, ``__anomaly__``).
    2. Remove ALL consecutive duplicate API calls (strict deduplication).
    3. Truncate to the first ``max_calls`` (default 100) calls.

    Args:
        sequence: Raw API call sequence.
        max_calls: Maximum number of calls to keep after deduplication.

    Returns:
        Preprocessed sequence of at most ``max_calls`` unique-consecutive
        API calls.
    """
    seq = remove_sandbox_tokens(sequence)
    seq = deduplicate_consecutive(seq)
    return seq[:max_calls]


def olivera_preprocess_samples(
    samples: List[Sample],
    max_calls: int = cfg.OLIVERA_SEQ_COLUMNS,
) -> List[Sample]:
    """Apply Olivera-style preprocessing to all samples.

    Args:
        samples: List of sample dicts with ``sequence`` key.
        max_calls: Maximum calls per sequence after deduplication.

    Returns:
        New list of sample dicts with Olivera-preprocessed sequences.
    """
    processed = []
    for s in samples:
        new_sample = dict(s)
        new_sample["sequence"] = olivera_style_preprocess(
            s["sequence"], max_calls,
        )
        processed.append(new_sample)
    logger.info(
        "Olivera-preprocessed %d samples (max %d calls).",
        len(processed), max_calls,
    )
    return processed


# ── Train/test split ──────────────────────────────────────────────────────


def stratified_split(
    samples: List[Sample],
    test_size: float = cfg.TEST_SIZE,
    random_seed: int = cfg.RANDOM_SEED,
) -> Tuple[List[Sample], List[Sample]]:
    """Stratified train/test split preserving class proportions.

    Args:
        samples: List of sample dicts with ``label`` key.
        test_size: Fraction of data for the test set.
        random_seed: Random seed for reproducibility.

    Returns:
        (train_samples, test_samples) tuple.
    """
    labels = [s["label"] for s in samples]
    train, test = train_test_split(
        samples,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    logger.info(
        "Split: %d train, %d test (%.0f%% test).",
        len(train),
        len(test),
        100 * len(test) / len(samples),
    )
    return train, test


# ── Vocabulary ─────────────────────────────────────────────────────────────


def build_vocabulary(train_samples: List[Sample]) -> Dict[str, int]:
    """Build a token→index vocabulary from training sequences.

    Index 0 is reserved for ``<PAD>`` and index 1 for ``<UNK>``.
    Remaining tokens are sorted alphabetically for determinism.

    Args:
        train_samples: Training-split samples with cleaned ``sequence`` fields.

    Returns:
        Dict mapping token strings to integer indices.
    """
    tokens: set = set()
    for s in train_samples:
        tokens.update(s["sequence"])

    vocab: Dict[str, int] = {
        cfg.PAD_TOKEN: cfg.PAD_INDEX,
        cfg.UNK_TOKEN: cfg.UNK_INDEX,
    }
    for idx, tok in enumerate(sorted(tokens), start=2):
        vocab[tok] = idx

    logger.info(
        "Vocabulary built: %d tokens (including PAD and UNK).", len(vocab)
    )
    return vocab


def save_vocabulary(vocab: Dict[str, int], path=cfg.VOCABULARY_PATH) -> None:
    """Save vocabulary to JSON."""
    save_json(vocab, path)
    logger.info("Vocabulary saved to %s.", path)


def load_vocabulary(path=cfg.VOCABULARY_PATH) -> Dict[str, int]:
    """Load vocabulary from JSON."""
    return load_json(path)


# ── Sequence encoding ─────────────────────────────────────────────────────


def encode_sequence(
    sequence: List[str],
    vocab: Dict[str, int],
) -> List[int]:
    """Encode a single API call sequence to integer indices.

    Tokens not in *vocab* are mapped to ``<UNK>`` (index 1).

    Args:
        sequence: Cleaned API call sequence.
        vocab: Token-to-index mapping.

    Returns:
        List of integer indices.
    """
    unk = vocab[cfg.UNK_TOKEN]
    return [vocab.get(tok, unk) for tok in sequence]


def encode_samples(
    samples: List[Sample],
    vocab: Dict[str, int],
) -> List[Sample]:
    """Encode sequences for all samples (adds ``encoded`` field).

    Args:
        samples: Cleaned samples with ``sequence`` key.
        vocab: Token-to-index mapping.

    Returns:
        New list of sample dicts with added ``encoded`` key.
    """
    encoded = []
    for s in samples:
        new_sample = dict(s)
        new_sample["encoded"] = encode_sequence(s["sequence"], vocab)
        encoded.append(new_sample)
    logger.info("Encoded %d samples.", len(encoded))
    return encoded


def pad_sequences(
    encoded_seqs: List[List[int]],
    max_len: int,
    pad_value: int = cfg.PAD_INDEX,
) -> np.ndarray:
    """Pad/truncate encoded sequences to a fixed length.

    Args:
        encoded_seqs: List of integer-encoded sequences.
        max_len: Target sequence length.
        pad_value: Value used for padding (default: PAD index 0).

    Returns:
        2-D numpy array of shape ``(n_samples, max_len)``.
    """
    n = len(encoded_seqs)
    result = np.full((n, max_len), pad_value, dtype=np.int32)
    for i, seq in enumerate(encoded_seqs):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length]
    return result


# ── WinMET → Mal-API-2019 format conversion ──────────────────────────────


def _winmet_parquet_to_mal_api_files(
    parquet_path,
    sequences_out,
    labels_out,
    drop_trojan: bool,
) -> int:
    """Convert the WinMET extraction Parquet into Mal-API-2019 file format.

    The WinMET extraction (``src/data_loading/extract_winmet.py``) produces
    a Parquet file with rich metadata (sha256, family_avclass, primary_class,
    etc.).  For cross-dataset generalizability evaluation, the test set must
    be in the *same* file format as Mal-API-2019 so that
    :func:`src.data_loading.data_loader.load_mal_api` can read it directly:

    * ``winmet_sequences.txt`` — one space-separated, lowercase API call
      sequence per line (matches ``data/Mal API.txt``)
    * ``winmet_labels.csv`` — one behavioral class label per line, no header
      (matches ``data/Mal API Labels.csv``)

    The behavioral class written to the labels file is the WinMET
    ``primary_class`` field — i.e. the AVClass family name (e.g. ``redline``,
    ``berbew``) has already been *replaced* with one of the Mal-API target
    classes (``Spyware``, ``Backdoor``, ``Downloader``, ``Worms``, ``Virus``,
    ``Trojan``) via the ``FAMILY_CLASS_MAPPING`` dict in the extraction
    script.  Rows whose ``primary_class`` is null/empty (no mapping) are
    dropped.

    Args:
        parquet_path: Path to ``winmet_extracted.parquet``.
        sequences_out: Output path for the Mal-API-format sequences ``.txt``.
        labels_out: Output path for the Mal-API-format labels ``.csv``.
        drop_trojan: If True, also drop rows whose ``primary_class`` is
            ``"Trojan"`` (used to match the no-Trojan trained models).

    Returns:
        The number of rows written to the output files.
    """
    # Local import keeps pyarrow as an optional dep for the rest of the module.
    import pyarrow.parquet as pq

    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    before = len(df)
    df = df[df["primary_class"].notna() & (df["primary_class"] != "")]
    if drop_trojan:
        df = df[df["primary_class"] != "Trojan"]
    logger.info(
        "WinMET → Mal-API conversion: kept %d/%d rows (drop_trojan=%s).",
        len(df), before, drop_trojan,
    )

    sequences_out = Path(sequences_out)
    labels_out = Path(labels_out)
    sequences_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    with open(sequences_out, "w", encoding="utf-8") as fseq, \
         open(labels_out, "w", encoding="utf-8") as flbl:
        for seq, lbl in zip(df["api_sequence"], df["primary_class"]):
            # api_sequence is already a space-joined lowercase string.
            fseq.write(seq + "\n")
            flbl.write(lbl + "\n")

    logger.info(
        "WinMET sequences written: %s (%d lines)", sequences_out, len(df),
    )
    logger.info(
        "WinMET labels written:    %s (%d lines)", labels_out, len(df),
    )
    return len(df)


def winmet_to_mal_api_format(
    parquet_path=cfg.WINMET_PARQUET_PATH,
    sequences_out=cfg.WINMET_SEQUENCES_PATH,
    labels_out=cfg.WINMET_LABELS_PATH,
) -> int:
    """Write the **full** WinMET dataset in Mal-API-2019 file format.

    This converts ``data/winmet/winmet_extracted.parquet`` into a pair of
    files (``winmet_sequences.txt`` + ``winmet_labels.csv``) that can be
    loaded directly by
    :func:`src.data_loading.data_loader.load_mal_api`, allowing WinMET to
    be used as the test set in the existing generalizability evaluation
    pipeline.

    The AVClass family names (e.g. ``redline``) are replaced with the
    behavioral class label expected by the Mal-API-2019 trained models
    (e.g. ``Spyware``).  Rows missing a class mapping are dropped.

    Use this variant when evaluating against models that *include* the
    ``Trojan`` class.

    Args:
        parquet_path: Path to the WinMET extraction Parquet.
        sequences_out: Output path for the sequences ``.txt`` file.
        labels_out: Output path for the labels ``.csv`` file.

    Returns:
        Number of rows written.
    """
    return _winmet_parquet_to_mal_api_files(
        parquet_path=parquet_path,
        sequences_out=sequences_out,
        labels_out=labels_out,
        drop_trojan=False,
    )


def winmet_to_mal_api_format_no_trojan(
    parquet_path=cfg.WINMET_PARQUET_PATH,
    sequences_out=cfg.WINMET_NO_TROJAN_SEQUENCES_PATH,
    labels_out=cfg.WINMET_NO_TROJAN_LABELS_PATH,
) -> int:
    """Write the WinMET dataset in Mal-API-2019 format **with Trojan removed**.

    Identical to :func:`winmet_to_mal_api_format` (replaces AVClass family
    names with Mal-API behavioral classes and matches the loader format),
    but additionally drops every sample whose ``primary_class`` is
    ``"Trojan"``.

    Use this variant when evaluating against the current
    ``*_no_trojan`` models in ``models/`` — those models were trained on
    the 6 non-Trojan classes only, so leaving Trojan samples in would
    introduce a class the model has never seen.

    Args:
        parquet_path: Path to the WinMET extraction Parquet.
        sequences_out: Output path for the no-Trojan sequences ``.txt``.
        labels_out: Output path for the no-Trojan labels ``.csv``.

    Returns:
        Number of rows written (Trojan rows excluded).
    """
    return _winmet_parquet_to_mal_api_files(
        parquet_path=parquet_path,
        sequences_out=sequences_out,
        labels_out=labels_out,
        drop_trojan=True,
    )


# ── MalbehavD-V1 preprocessing ────────────────────────────────────────────


def preprocess_malbehavd_sequences(
    samples: List[Sample],
    vocab: Dict[str, int],
    max_repeats: int = cfg.MAX_CONSECUTIVE_DUPLICATES,
) -> List[Sample]:
    """Preprocess MalbehavD-V1 samples for inference.

    Steps: lowercase all API names, clean, encode with the training vocabulary
    (unmapped calls become ``<UNK>``).

    Args:
        samples: Raw MalbehavD-V1 samples.
        vocab: Training vocabulary.
        max_repeats: Max consecutive duplicates.

    Returns:
        Preprocessed samples with ``sequence`` (lowercased+cleaned) and
        ``encoded`` fields.
    """
    processed = []
    for s in samples:
        new_sample = dict(s)
        lowered = [tok.lower() for tok in s["sequence"]]
        cleaned = clean_sequence(lowered, max_repeats)
        new_sample["sequence"] = cleaned
        new_sample["encoded"] = encode_sequence(cleaned, vocab)
        processed.append(new_sample)
    logger.info("Preprocessed %d MalbehavD-V1 samples.", len(processed))
    return processed


def compute_unk_ratio(samples: List[Sample], vocab: Dict[str, int]) -> float:
    """Compute the fraction of tokens that map to ``<UNK>`` across all samples.

    Args:
        samples: Samples with ``sequence`` field (already lowercased).
        vocab: Training vocabulary.

    Returns:
        Float in [0, 1] representing the unknown-token ratio.
    """
    total = 0
    unknown = 0
    for s in samples:
        for tok in s["sequence"]:
            total += 1
            if tok not in vocab:
                unknown += 1
    return unknown / total if total > 0 else 0.0


# ── Full pipeline ─────────────────────────────────────────────────────────


def run_preprocessing_pipeline(
    samples: List[Sample],
    cache_train_path=cfg.PREPROCESSED_TRAIN_PATH,
    cache_test_path=cfg.PREPROCESSED_TEST_PATH,
    vocab_path=cfg.VOCABULARY_PATH,
    use_cache: bool = True,
) -> Tuple[List[Sample], List[Sample], Dict[str, int]]:
    """Run the full preprocessing pipeline on Mal-API-2019 data.

    1. Clean sequences
    2. Stratified 80/20 split
    3. Build vocabulary from training split
    4. Encode sequences

    Caches results to disk.  On subsequent calls with ``use_cache=True``,
    loads from cache if available.

    Args:
        samples: Raw Mal-API-2019 samples.
        cache_train_path: Path to cache preprocessed training data.
        cache_test_path: Path to cache preprocessed test data.
        vocab_path: Path to save/load vocabulary JSON.
        use_cache: Whether to use cached results if available.

    Returns:
        (train_samples, test_samples, vocab) — samples have ``sequence``,
        ``label``, and ``encoded`` fields.
    """
    if (
        use_cache
        and cache_train_path.exists()
        and cache_test_path.exists()
        and vocab_path.exists()
    ):
        logger.info("Loading preprocessed data from cache.")
        train = load_pickle(cache_train_path)
        test = load_pickle(cache_test_path)
        vocab = load_json(vocab_path)
        return train, test, vocab

    # 1. Clean
    cleaned = clean_samples(samples)

    # 2. Split
    train, test = stratified_split(cleaned)

    # 3. Vocabulary (from training data only)
    vocab = build_vocabulary(train)
    save_vocabulary(vocab, vocab_path)

    # 4. Encode
    train = encode_samples(train, vocab)
    test = encode_samples(test, vocab)

    # Cache
    save_pickle(train, cache_train_path)
    save_pickle(test, cache_test_path)
    logger.info("Preprocessed data cached.")

    return train, test, vocab
