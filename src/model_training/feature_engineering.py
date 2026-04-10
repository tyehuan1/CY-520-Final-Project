"""
Feature engineering for XGBoost: length-normalized TF, statistical,
API category, and bigram transition features.

All feature sets are designed to be robust to sequence length variation
so that models trained on long traces (Mal-API, median ~368 calls) can
generalize to shorter traces (MalBehavD, median ~24 calls).

Key design choices:
  - TF-IDF replaced with L1-normalized term frequency (no IDF weighting).
    IDF inflates weights unpredictably on short sequences.
  - Statistical features use log-scaled counts instead of raw counts.
  - Category ratio features are already length-normalized (proportions).
  - Bigram transition features capture sequential patterns as proportions.
"""

import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer

import config as cfg
from src.data_loading.api_categories import CATEGORIES, get_category
from src.utils import get_logger, load_pickle, save_pickle

logger = get_logger(__name__)

Sample = Dict[str, object]


# ── Length-normalized TF features ────────────────────────────────────────────


def build_tfidf_vectorizer(
    train_samples: List[Sample],
    max_features: int = cfg.TFIDF_MAX_FEATURES,
    ngram_range: Tuple[int, int] = cfg.TFIDF_NGRAM_RANGE,
) -> TfidfVectorizer:
    """Fit a length-normalized TF vectorizer on training sequences.

    Uses ``use_idf=False`` and ``norm='l1'`` so that feature vectors sum
    to 1.0 regardless of sequence length.  This makes a 24-call trace
    directly comparable to a 368-call trace.

    Args:
        train_samples: Training samples with ``sequence`` field.
        max_features: Maximum number of n-gram features.
        ngram_range: N-gram range (min_n, max_n).

    Returns:
        Fitted :class:`TfidfVectorizer` (with IDF disabled).
    """
    corpus = [" ".join(s["sequence"]) for s in train_samples]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer="word",
        use_idf=False,
        norm="l1",
    )
    vectorizer.fit(corpus)
    logger.info(
        "TF vectorizer fitted: %d features (ngram_range=%s, norm=l1, no IDF).",
        len(vectorizer.vocabulary_),
        ngram_range,
    )
    return vectorizer


def tfidf_transform(
    samples: List[Sample],
    vectorizer: TfidfVectorizer,
) -> np.ndarray:
    """Transform samples into length-normalized TF feature matrix.

    Args:
        samples: Samples with ``sequence`` field.
        vectorizer: Fitted TF vectorizer.

    Returns:
        Dense numpy array of shape ``(n_samples, n_features)``.
    """
    corpus = [" ".join(s["sequence"]) for s in samples]
    matrix = vectorizer.transform(corpus)
    if issparse(matrix):
        matrix = matrix.toarray()
    return matrix


# ── Statistical / sequential features ──────────────────────────────────────


def compute_statistical_features(
    samples: List[Sample],
    top_k: int = cfg.TOP_K_API_FREQUENCIES,
    log_dampen: bool = False,
) -> np.ndarray:
    """Compute length-robust statistical features for each sample.

    Features per sample:
    1. log2(sequence_length + 1) — log-scaled to compress range
    2. log2(unique_count + 1) — log-scaled
    3. Unique-to-total ratio
    4-8. Top-k API call frequencies; zero-padded if < k unique
    9. Shannon entropy of the API call frequency distribution

    When ``log_dampen=False`` (legacy), features 4-8 are raw ratios
    ``count / total``.  When ``log_dampen=True`` (V2), they become
    ``log2(count + 1) / log2(total + 1)`` which compresses the range so
    a CAPE trace with 70% of one token (~0.95 after log) is much closer
    to a Mal-API trace with 20% of one token (~0.80 after log) than the
    raw ratios would suggest (0.70 vs 0.20).

    Args:
        samples: Samples with ``sequence`` field.
        top_k: Number of top API call frequency ratios to include.
        log_dampen: If True, log-scale the top-k frequency ratios.

    Returns:
        Array of shape ``(n_samples, 3 + top_k + 1)``.
    """
    n_features = 3 + top_k + 1  # log_length, log_unique, ratio, top_k ratios, entropy
    features = np.zeros((len(samples), n_features), dtype=np.float64)

    for i, sample in enumerate(samples):
        seq = sample["sequence"]
        total = len(seq)

        if total == 0:
            continue

        counts = Counter(seq)
        unique = len(counts)
        ratio = unique / total

        # Top-k frequency ratios (descending)
        sorted_counts = sorted(counts.values(), reverse=True)
        if log_dampen:
            log_total = math.log2(total + 1)
            top_k_ratios = [
                math.log2(c + 1) / log_total for c in sorted_counts[:top_k]
            ]
        else:
            top_k_ratios = [c / total for c in sorted_counts[:top_k]]
        # Pad if fewer than top_k unique calls
        top_k_ratios += [0.0] * (top_k - len(top_k_ratios))

        # Shannon entropy
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        features[i, 0] = math.log2(total + 1)
        features[i, 1] = math.log2(unique + 1)
        features[i, 2] = ratio
        features[i, 3 : 3 + top_k] = top_k_ratios
        features[i, 3 + top_k] = entropy

    logger.info(
        "Statistical features computed: shape %s (log_dampen=%s).",
        features.shape, log_dampen,
    )
    return features


STATISTICAL_FEATURE_NAMES = (
    ["log_seq_length", "log_unique_count", "unique_ratio"]
    + [f"top{k+1}_freq_ratio" for k in range(cfg.TOP_K_API_FREQUENCIES)]
    + ["shannon_entropy"]
)


# ── API category ratio features ──────────────────────────────────────────────


def compute_category_features(
    samples: List[Sample],
    log_dampen: bool = False,
) -> np.ndarray:
    """Compute behavioral category ratio features for each sample.

    By default, counts how many API calls fall into each category and
    divides by total sequence length to produce a ratio vector — already
    length-invariant, but vulnerable to a single high-frequency token
    dominating its category (e.g. CAPE traces with 600k+ ``closehandle``
    calls collapse the file-handle category to a single value).

    Set ``log_dampen=True`` to weight each *unique* token by ``log2(count + 1)``
    instead of its raw count.  Repeated tokens contribute logarithmically
    rather than linearly, which is the cross-sandbox-robust variant used
    by the V2 training pipeline.

    Args:
        samples: Samples with ``sequence`` field.
        log_dampen: If True, use log-weighted unique-token counts instead
            of raw token counts before length-normalizing.

    Returns:
        Array of shape ``(n_samples, len(CATEGORIES))``.
    """
    cat_index = {cat: idx for idx, cat in enumerate(CATEGORIES)}
    n_cats = len(CATEGORIES)
    features = np.zeros((len(samples), n_cats), dtype=np.float64)

    for i, sample in enumerate(samples):
        seq: List[str] = sample["sequence"]  # type: ignore[assignment]
        total = len(seq)
        if total == 0:
            continue

        if log_dampen:
            counts = Counter(seq)
            weighted_total = 0.0
            for tok, c in counts.items():
                w = math.log2(c + 1)
                features[i, cat_index[get_category(tok)]] += w
                weighted_total += w
            if weighted_total > 0:
                features[i] /= weighted_total
        else:
            for tok in seq:
                features[i, cat_index[get_category(tok)]] += 1
            features[i] /= total

    logger.info(
        "Category features computed: shape %s (log_dampen=%s).",
        features.shape, log_dampen,
    )
    return features


CATEGORY_FEATURE_NAMES = [f"cat_{cat}" for cat in CATEGORIES]


# ── Bigram transition features ───────────────────────────────────────────────


def compute_bigram_transition_features(
    samples: List[Sample],
    log_dampen: bool = False,
) -> np.ndarray:
    """Compute API category bigram transition proportions.

    By default each consecutive pair contributes 1 and the result is
    divided by ``len(seq) - 1`` — already length-invariant but dominated
    by repeated transitions in CAPE-style traces.

    With ``log_dampen=True``, distinct token bigrams are weighted by
    ``log2(count_in_seq + 1)`` and normalized by the sum of those weights,
    so a 600k-occurrence ``closehandle→closehandle`` transition contributes
    log2(600001) ≈ 19.2 instead of 600k.

    Args:
        samples: Samples with ``sequence`` field.
        log_dampen: If True, use log-weighted unique-bigram counts.

    Returns:
        Array of shape ``(n_samples, len(CATEGORIES)**2)``.
    """
    n_cats = len(CATEGORIES)
    cat_index = {cat: idx for idx, cat in enumerate(CATEGORIES)}
    features = np.zeros((len(samples), n_cats * n_cats), dtype=np.float64)

    for i, sample in enumerate(samples):
        seq: List[str] = sample["sequence"]  # type: ignore[assignment]
        if len(seq) < 2:
            continue

        if log_dampen:
            bigram_counts: Counter = Counter(
                (seq[j], seq[j + 1]) for j in range(len(seq) - 1)
            )
            weighted_total = 0.0
            for (a, b), c in bigram_counts.items():
                w = math.log2(c + 1)
                src = cat_index[get_category(a)]
                dst = cat_index[get_category(b)]
                features[i, src * n_cats + dst] += w
                weighted_total += w
            if weighted_total > 0:
                features[i] /= weighted_total
        else:
            n_transitions = len(seq) - 1
            for j in range(n_transitions):
                src = cat_index[get_category(seq[j])]
                dst = cat_index[get_category(seq[j + 1])]
                features[i, src * n_cats + dst] += 1
            features[i] /= n_transitions

    logger.info(
        "Bigram transition features computed: shape %s (log_dampen=%s).",
        features.shape, log_dampen,
    )
    return features


BIGRAM_FEATURE_NAMES = [
    f"bigram_{src}_{dst}" for src in CATEGORIES for dst in CATEGORIES
]


# ── Combined feature matrix ─────────────────────────────────────────────────


def build_feature_matrix(
    samples: List[Sample],
    tfidf_vectorizer: TfidfVectorizer,
    top_k: int = cfg.TOP_K_API_FREQUENCIES,
    log_dampen: bool = False,
) -> np.ndarray:
    """Build the full concatenated feature matrix for XGBoost.

    Concatenates:
    1. Length-normalized TF features (n_samples x <=5000)
    2. Statistical features (n_samples x 9)
    3. API category ratio features (n_samples x 8)
    4. Bigram transition features (n_samples x 64)

    Args:
        samples: Samples with ``sequence`` field.
        tfidf_vectorizer: Fitted TF vectorizer.
        top_k: Number of top-k frequency features.
        log_dampen: If True, propagate log-dampened weighting to the
            category and bigram features.  V2 models train and evaluate
            with this set to True so a single 600k-call CAPE token can't
            dominate the proportion vector.

    Returns:
        Dense array of shape ``(n_samples, total_features)``.
    """
    parts = []
    part_names = []

    tf = tfidf_transform(samples, tfidf_vectorizer)
    parts.append(tf)
    part_names.append(f"tf={tf.shape[1]}")

    stats = compute_statistical_features(samples, top_k, log_dampen=log_dampen)
    parts.append(stats)
    part_names.append(f"stats={stats.shape[1]}")

    cats = compute_category_features(samples, log_dampen=log_dampen)
    parts.append(cats)
    part_names.append(f"cats={cats.shape[1]}")

    bigrams = compute_bigram_transition_features(samples, log_dampen=log_dampen)
    parts.append(bigrams)
    part_names.append(f"bigrams={bigrams.shape[1]}")

    combined = np.hstack(parts)
    logger.info(
        "Combined feature matrix: %s (%s).",
        combined.shape,
        ", ".join(part_names),
    )
    return combined


def get_feature_names(
    tfidf_vectorizer: TfidfVectorizer,
) -> List[str]:
    """Return ordered list of all feature names matching the combined matrix.

    Args:
        tfidf_vectorizer: Fitted TF vectorizer.

    Returns:
        List of feature name strings.
    """
    names = tfidf_vectorizer.get_feature_names_out().tolist()
    return (
        names
        + STATISTICAL_FEATURE_NAMES
        + CATEGORY_FEATURE_NAMES
        + BIGRAM_FEATURE_NAMES
    )
