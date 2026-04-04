"""
Feature engineering for XGBoost: TF-IDF, statistical, and API category features.

All three feature sets are computed independently, then concatenated into a
single dense matrix for model training.
"""

import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer

import config as cfg
from src.api_categories import CATEGORIES, get_category
from src.utils import get_logger, load_pickle, save_pickle

logger = get_logger(__name__)

Sample = Dict[str, object]


# ── TF-IDF features ───────────────────────────────────────────────────────


def build_tfidf_vectorizer(
    train_samples: List[Sample],
    max_features: int = cfg.TFIDF_MAX_FEATURES,
    ngram_range: Tuple[int, int] = cfg.TFIDF_NGRAM_RANGE,
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on training sequences.

    Args:
        train_samples: Training samples with ``sequence`` field.
        max_features: Maximum number of TF-IDF features.
        ngram_range: N-gram range (min_n, max_n).

    Returns:
        Fitted :class:`TfidfVectorizer`.
    """
    corpus = [" ".join(s["sequence"]) for s in train_samples]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer="word",
    )
    vectorizer.fit(corpus)
    logger.info(
        "TF-IDF vectorizer fitted: %d features (ngram_range=%s).",
        len(vectorizer.vocabulary_),
        ngram_range,
    )
    return vectorizer


def tfidf_transform(
    samples: List[Sample],
    vectorizer: TfidfVectorizer,
) -> np.ndarray:
    """Transform samples into TF-IDF feature matrix.

    Args:
        samples: Samples with ``sequence`` field.
        vectorizer: Fitted TF-IDF vectorizer.

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
) -> np.ndarray:
    """Compute statistical features for each sample.

    Features per sample:
    1. Sequence length (total API calls)
    2. Unique API call count
    3. Unique-to-total ratio
    4-8. Top-k API call frequencies (as ratios); zero-padded if < k unique
    9. Shannon entropy of the API call frequency distribution

    Args:
        samples: Samples with ``sequence`` field.
        top_k: Number of top API call frequency ratios to include.

    Returns:
        Array of shape ``(n_samples, 3 + top_k + 1)``.
    """
    n_features = 3 + top_k + 1  # length, unique, ratio, top_k ratios, entropy
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
        top_k_ratios = [c / total for c in sorted_counts[:top_k]]
        # Pad if fewer than top_k unique calls
        top_k_ratios += [0.0] * (top_k - len(top_k_ratios))

        # Shannon entropy
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        features[i, 0] = total
        features[i, 1] = unique
        features[i, 2] = ratio
        features[i, 3 : 3 + top_k] = top_k_ratios
        features[i, 3 + top_k] = entropy

    logger.info("Statistical features computed: shape %s.", features.shape)
    return features


STATISTICAL_FEATURE_NAMES = (
    ["seq_length", "unique_count", "unique_ratio"]
    + [f"top{k+1}_freq_ratio" for k in range(cfg.TOP_K_API_FREQUENCIES)]
    + ["shannon_entropy"]
)


# ── API category ratio features ────────────────────────────────────────────


def compute_category_features(samples: List[Sample]) -> np.ndarray:
    """Compute behavioral category ratio features for each sample.

    For each sample, counts how many API calls fall into each category and
    divides by total sequence length to produce a ratio vector.

    Args:
        samples: Samples with ``sequence`` field.

    Returns:
        Array of shape ``(n_samples, len(CATEGORIES))``.
    """
    cat_index = {cat: idx for idx, cat in enumerate(CATEGORIES)}
    n_cats = len(CATEGORIES)
    features = np.zeros((len(samples), n_cats), dtype=np.float64)

    for i, sample in enumerate(samples):
        seq = sample["sequence"]
        total = len(seq)
        if total == 0:
            continue
        for tok in seq:
            cat = get_category(tok)
            features[i, cat_index[cat]] += 1
        features[i] /= total

    logger.info("Category features computed: shape %s.", features.shape)
    return features


CATEGORY_FEATURE_NAMES = [f"cat_{cat}" for cat in CATEGORIES]


# ── Combined feature matrix ───────────────────────────────────────────────


def build_feature_matrix(
    samples: List[Sample],
    tfidf_vectorizer: TfidfVectorizer,
    top_k: int = cfg.TOP_K_API_FREQUENCIES,
) -> np.ndarray:
    """Build the full concatenated feature matrix for XGBoost.

    Concatenates:
    1. TF-IDF features (n_samples x <=5000)
    2. Statistical features (n_samples x 9)
    3. API category ratio features (n_samples x 8)

    Args:
        samples: Samples with ``sequence`` field.
        tfidf_vectorizer: Fitted TF-IDF vectorizer.
        top_k: Number of top-k frequency features.

    Returns:
        Dense array of shape ``(n_samples, total_features)``.
    """
    parts = []
    part_names = []

    tfidf = tfidf_transform(samples, tfidf_vectorizer)
    parts.append(tfidf)
    part_names.append(f"tfidf={tfidf.shape[1]}")

    stats = compute_statistical_features(samples, top_k)
    parts.append(stats)
    part_names.append(f"stats={stats.shape[1]}")

    cats = compute_category_features(samples)
    parts.append(cats)
    part_names.append(f"cats={cats.shape[1]}")

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
        tfidf_vectorizer: Fitted TF-IDF vectorizer.

    Returns:
        List of feature name strings.
    """
    names = tfidf_vectorizer.get_feature_names_out().tolist()
    return (
        names
        + STATISTICAL_FEATURE_NAMES
        + CATEGORY_FEATURE_NAMES
    )
