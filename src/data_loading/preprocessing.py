"""
Preprocessing pipeline: cleaning, vocabulary building, encoding, splitting.

All transformations are deterministic given the random seed in config.
The vocabulary is built exclusively from the training split.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def load_winmet_samples(
    parquet_path=cfg.WINMET_PARQUET_PATH,
    drop_trojan: bool = True,
    max_seq_len=cfg.MAX_RAW_SEQUENCE_LENGTH,
) -> List[Sample]:
    """Load WinMET samples from the extraction Parquet for in-memory analysis.

    Returns sample dicts in the same shape as
    :func:`src.data_loading.data_loader.load_mal_api`'s output (``sequence``
    + ``label``), but with two extra fields used by the generalizability
    analysis:

    * ``family_avclass``: the original AVClass family name (e.g. ``redline``,
      ``vbclone``).  Needed to look up secondary behavioral classes via
      ``FAMILY_CLASS_MAPPING`` in ``extract_winmet.py`` for the
      misclassification analysis (e.g. asking "did this prediction land
      on a *secondary* class for this family?").
    * ``sha256``: the file hash, useful when joining back to the parquet.

    Use this function (rather than the ``.txt`` / ``.csv`` files written
    by :func:`winmet_to_mal_api_format`) when you need the parquet's
    metadata in addition to the sequence and label.  The ``.txt``/``.csv``
    files are still the right choice when you only need to feed WinMET
    through code that already speaks the Mal-API-2019 file format.

    Args:
        parquet_path: Path to ``winmet_extracted.parquet``.
        drop_trojan: If True, drop rows whose ``primary_class`` is
            ``"Trojan"`` (matches the no-Trojan trained models).
        max_seq_len: If set, truncate sequences to this many tokens (matches
            the truncation used by ``load_mal_api`` so the data goes through
            the rest of the pipeline at the same scale).

    Returns:
        List of sample dicts with keys ``sequence``, ``label``,
        ``family_avclass``, ``sha256``.
    """
    import pyarrow.parquet as pq

    # Read only the columns we need; avoid going through pandas because the
    # api_sequence column contains very large strings (sequences up to ~250k
    # tokens) and pandas' pyarrow-backed string take/filter ops blow out
    # memory when filtering rows.  We filter and iterate at the pyarrow level
    # using Python lists.
    columns = [
        "sha256", "family_avclass", "primary_class",
        "secondary_classes", "api_sequence",
    ]
    table = pq.read_table(str(parquet_path), columns=columns)
    before = table.num_rows

    primary = table.column("primary_class").to_pylist()
    family = table.column("family_avclass").to_pylist()
    sha = table.column("sha256").to_pylist()
    secondary_col = table.column("secondary_classes").to_pylist()
    api_seq = table.column("api_sequence").to_pylist()
    del table  # release the arrow buffer once we have python lists

    samples: List[Sample] = []
    for i in range(before):
        pc = primary[i]
        if pc is None or pc == "":
            continue
        if drop_trojan and pc == "Trojan":
            continue

        seq_str = api_seq[i] or ""
        seq = seq_str.split()
        if max_seq_len is not None and len(seq) > max_seq_len:
            seq = seq[:max_seq_len]

        secondary = secondary_col[i]
        if secondary is None:
            secondary = []
        else:
            secondary = [str(c) for c in list(secondary)]

        samples.append({
            "sequence": seq,
            "label": pc,
            "family_avclass": family[i],
            "secondary_classes": secondary,
            "sha256": sha[i],
        })

    logger.info(
        "load_winmet_samples: kept %d/%d rows (drop_trojan=%s).",
        len(samples), before, drop_trojan,
    )
    return samples


# ── Cross-sandbox token normalization ────────────────────────────────────
#
# The Mal-API-2019 vocabulary is built from Cuckoo-hooked API names (only
# ~265 unique calls — Cuckoo only hooks at the NT layer).  Datasets like
# WinMET come from CAPE, which hooks both Win32 and lower-level routines,
# so the same behavior shows up under different names.  These rules let us
# rewrite CAPE-flavored tokens into their Mal-API equivalents *before*
# vocabulary lookup, recovering hits that would otherwise become <UNK>.
#
# Three categories:
#   A) Pure spelling/suffix differences (process32next vs process32nextw,
#      ldrgetprocedureaddressforcaller vs ldrgetprocedureaddress, ...).
#      Handled by suffix-stripping plus W/A-suffix retry against the vocab.
#   B) Win32 ↔ NT layer aliases (sleep ↔ ntdelayexecution, virtualalloc ↔
#      ntallocatevirtualmemory, ...).  Handled by an explicit alias table
#      below — the user verified each entry against the vocabulary.
#   C) Calls Cuckoo never hooks at all (getprocessheap, ntwaitforsingleobject,
#      ntqueryinformationtoken, ...).  These stay <UNK>.

# Win32 / kernel-mode aliases.  Each value MUST be a token that exists in the
# Mal-API-2019 vocabulary; otherwise the rewrite is a no-op.
WIN32_TO_NT_ALIASES: Dict[str, str] = {
    # ── Memory ──
    "virtualalloc":           "ntallocatevirtualmemory",
    "virtualallocex":         "ntallocatevirtualmemory",
    "virtualfree":            "ntfreevirtualmemory",
    "virtualfreeex":          "ntfreevirtualmemory",
    "virtualprotect":         "ntprotectvirtualmemory",
    "virtualprotectex":       "ntprotectvirtualmemory",
    "globalalloc":            "ntallocatevirtualmemory",
    "localalloc":             "ntallocatevirtualmemory",
    "heapalloc":              "ntallocatevirtualmemory",
    # ── Threading / waits ──
    "sleep":                  "ntdelayexecution",
    "sleepex":                "ntdelayexecution",
    # ── Handles ──
    "closehandle":            "ntclose",
    # ── Files ──
    "createfilew":            "ntcreatefile",
    "createfilea":            "ntcreatefile",
    "openfile":               "ntopenfile",
    "readfile":               "ntreadfile",
    "readfileex":             "ntreadfile",
    "writefile":              "ntwritefile",
    "writefileex":            "ntwritefile",
    # ── Processes ──
    "createprocessw":         "createprocessinternalw",
    "createprocessa":         "createprocessinternalw",
    "createprocessasuserw":   "createprocessinternalw",
    "createprocessasusera":   "createprocessinternalw",
    "ntcreateuserprocess":    "createprocessinternalw",
    # ── Modules ──
    "loadlibraryw":           "ldrloaddll",
    "loadlibrarya":           "ldrloaddll",
    "loadlibraryexw":         "ldrloaddll",
    "loadlibraryexa":         "ldrloaddll",
    "getmodulehandlew":       "ldrgetdllhandle",
    "getmodulehandlea":       "ldrgetdllhandle",
    "getmodulehandleexw":     "ldrgetdllhandle",
    "getmodulehandleexa":     "ldrgetdllhandle",
    "getprocaddress":         "ldrgetprocedureaddress",
    # ── File-attribute info classes ──
    "ntqueryfullattributesfile": "ntqueryattributesfile",
    # ── Resources ──
    "lockresource":           "loadresource",
    # ── Process snapshots (already W-suffixed in vocab) ──
    "process32next":          "process32nextw",
    "process32first":         "process32firstw",
    "thread32next":           "thread32next",   # already in vocab
    "module32next":           "module32nextw",
    "module32first":          "module32firstw",
    # ── Process creation (legacy / wrapper variants) ──
    "winexec":                "createprocessinternalw",
    # ── File ops (transactional wrapper) ──
    "movefilewithprogresstransactedw": "movefilewithprogressw",
    "movefilewithprogresstransacteda": "movefilewithprogressw",
    "movefilew":              "movefilewithprogressw",
    "movefilea":              "movefilewithprogressw",
    "movefileexw":            "movefilewithprogressw",
    "movefileexa":            "movefilewithprogressw",
    # ── Time ──
    "getsystemtime":          "getsystemtimeasfiletime",
    "getlocaltime":           "getsystemtimeasfiletime",
}

# Suffixes that mean "the same call but the caller-tagged variant".
# Tried in order; the first stripped form that exists in the vocab wins.
_NORMALIZATION_SUFFIXES: Tuple[str, ...] = (
    "forcaller",   # ldrgetprocedureaddressforcaller -> ldrgetprocedureaddress
)


def normalize_token_for_vocab(tok: str, vocab: Dict[str, int]) -> str:
    """Try to rewrite a CAPE-flavored API token into a Mal-API vocab token.

    Returns the original token unchanged if no mapping recovers a vocab hit.
    The vocab is consulted at every step so we never produce a token that
    isn't actually a vocabulary entry.

    Order of operations (first hit wins):
      1. Token already in vocab → return as-is.
      2. Explicit Win32→NT alias table.
      3. Strip "forcaller" / similar suffixes.
      4. Append a ``w`` or ``a`` suffix (e.g. process32next → process32nextw).
      5. Strip a trailing ``ex`` (e.g. ntsetinformationprocessex → ...process)
         then retry steps 1+4.

    Args:
        tok: Lowercased API name from a sandbox trace.
        vocab: The training-set vocabulary dict.

    Returns:
        The normalized token (in vocab) or the original token (will become
        ``<UNK>`` downstream).
    """
    if tok in vocab:
        return tok

    aliased = WIN32_TO_NT_ALIASES.get(tok)
    if aliased is not None and aliased in vocab:
        return aliased

    for suf in _NORMALIZATION_SUFFIXES:
        if tok.endswith(suf):
            stripped = tok[: -len(suf)]
            if stripped in vocab:
                return stripped

    # Append W/A
    for suf in ("w", "a"):
        cand = tok + suf
        if cand in vocab:
            return cand

    # Strip trailing "ex" / "ex2" then retry vocab + W/A
    for ex_suf in ("ex2", "ex"):
        if tok.endswith(ex_suf):
            stripped = tok[: -len(ex_suf)]
            if stripped in vocab:
                return stripped
            for suf in ("w", "a"):
                if stripped + suf in vocab:
                    return stripped + suf

    return tok


def normalize_sequence_for_vocab(
    sequence: List[str], vocab: Dict[str, int],
) -> List[str]:
    """Apply :func:`normalize_token_for_vocab` to every token in a sequence."""
    return [normalize_token_for_vocab(t, vocab) for t in sequence]


# ── MalbehavD-V1 preprocessing ────────────────────────────────────────────


def preprocess_external_samples(
    samples: List[Sample],
    vocab: Dict[str, int],
    max_repeats: int = cfg.MAX_CONSECUTIVE_DUPLICATES,
    normalize_for_vocab: bool = False,
    dataset_name: str = "external",
) -> List[Sample]:
    """Preprocess samples from a non-training dataset (MalbehavD-V1, WinMET).

    Unified cleaning + (optional) cross-sandbox normalization + encoding so
    that every external dataset goes through the same path before being
    cached by ``build_no_trojan_dataset.py``.

    Steps: lowercase all API names, clean, optionally rewrite tokens via
    :func:`normalize_token_for_vocab` to recover cross-sandbox aliases,
    encode with the training vocabulary (anything still unmapped becomes
    ``<UNK>``).

    Args:
        samples: Raw samples.
        vocab: Training vocabulary.
        max_repeats: Max consecutive duplicates.
        normalize_for_vocab: If True, apply :func:`normalize_token_for_vocab`
            to every token after cleaning.  Use for cross-sandbox evaluation
            (e.g. WinMET) where the source sandbox emits API names under
            different aliases than Cuckoo (the source of the Mal-API vocab).
        dataset_name: Logged tag for traceability.

    Returns:
        Preprocessed samples with ``sequence`` (lowercased+cleaned, optionally
        normalized) and ``encoded`` fields.  All other fields on the input
        dicts are preserved.
    """
    processed = []
    for s in samples:
        new_sample = dict(s)
        lowered = [tok.lower() for tok in s["sequence"]]
        cleaned = clean_sequence(lowered, max_repeats)
        if normalize_for_vocab:
            cleaned = normalize_sequence_for_vocab(cleaned, vocab)
        new_sample["sequence"] = cleaned
        new_sample["encoded"] = encode_sequence(cleaned, vocab)
        processed.append(new_sample)
    logger.info("Preprocessed %d %s samples.", len(processed), dataset_name)
    return processed


# Backwards-compatible alias — older callers may still import the old name.
preprocess_malbehavd_sequences = preprocess_external_samples


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
