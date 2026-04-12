"""
Tests for the Olivera experiment dataset build pipeline.

Validates that the preprocessing produces correctly aligned training
and test artifacts: shared vocabulary, lowercased tokens, no artifacts,
proper encoding, and class filtering.
"""

import pytest
from collections import Counter

import config as cfg
from src.utils import load_json, load_pickle


@pytest.fixture(scope="module")
def train_samples():
    return load_pickle(cfg.OLIVERA_TRAIN_PATH)


@pytest.fixture(scope="module")
def test_samples():
    return load_pickle(cfg.OLIVERA_TEST_PATH)


@pytest.fixture(scope="module")
def olivera_samples():
    return load_pickle(cfg.OLIVERA_EXT_TEST_PATH)


@pytest.fixture(scope="module")
def vocab():
    return load_json(cfg.OLIVERA_VOCABULARY_PATH)


@pytest.fixture(scope="module")
def label_encoder():
    return load_pickle(cfg.OLIVERA_LABEL_ENCODER_PATH)


# ── Vocabulary ──────────────────────────────────────────────────────────


class TestVocabulary:
    """Validate the shared vocabulary."""

    def test_has_pad_and_unk(self, vocab):
        assert cfg.PAD_TOKEN in vocab
        assert cfg.UNK_TOKEN in vocab
        assert vocab[cfg.PAD_TOKEN] == cfg.PAD_INDEX
        assert vocab[cfg.UNK_TOKEN] == cfg.UNK_INDEX

    def test_all_lowercase(self, vocab):
        for token in vocab:
            if token in (cfg.PAD_TOKEN, cfg.UNK_TOKEN):
                continue
            assert token == token.lower(), f"Token '{token}' not lowercase"

    def test_no_sandbox_tokens(self, vocab):
        for token in vocab:
            assert not token.startswith("__"), f"Sandbox token '{token}' in vocab"

    def test_reasonable_size(self, vocab):
        # ~240 tokens expected (236 API + PAD + UNK)
        assert 200 < len(vocab) < 350


# ── Training data ───────────────────────────────────────────────────────


class TestTrainData:
    """Validate the Mal-API training split."""

    def test_has_encoded_field(self, train_samples):
        for s in train_samples[:50]:
            assert "encoded" in s
            assert isinstance(s["encoded"], list)
            assert len(s["encoded"]) == len(s["sequence"])

    def test_sequences_lowercased(self, train_samples):
        for s in train_samples[:100]:
            for tok in s["sequence"]:
                assert tok == tok.lower()

    def test_max_length_100(self, train_samples):
        for s in train_samples:
            assert len(s["sequence"]) <= cfg.OLIVERA_SEQ_COLUMNS

    def test_no_empty_sequences(self, train_samples):
        for s in train_samples:
            assert len(s["sequence"]) > 0

    def test_has_trojan(self, train_samples):
        labels = set(s["label"] for s in train_samples)
        assert "Trojan" in labels

    def test_eight_classes(self, label_encoder):
        assert len(label_encoder.classes_) == 8

    def test_encoding_uses_vocab(self, train_samples, vocab):
        unk_idx = vocab[cfg.UNK_TOKEN]
        for s in train_samples[:100]:
            for tok, enc in zip(s["sequence"], s["encoded"]):
                expected = vocab.get(tok, unk_idx)
                assert enc == expected, (
                    f"Token '{tok}' encoded as {enc}, expected {expected}"
                )


# ── Olivera test data ───────────────────────────────────────────────────


class TestOliveraData:
    """Validate the Olivera external test set."""

    def test_has_encoded_field(self, olivera_samples):
        for s in olivera_samples[:50]:
            assert "encoded" in s

    def test_sequences_lowercased(self, olivera_samples):
        for s in olivera_samples[:100]:
            for tok in s["sequence"]:
                assert tok == tok.lower()

    def test_no_sandbox_tokens(self, olivera_samples):
        for s in olivera_samples[:200]:
            for tok in s["sequence"]:
                assert not tok.startswith("__")

    def test_no_benign(self, olivera_samples):
        for s in olivera_samples:
            assert s["label"] != cfg.BENIGN_LABEL

    def test_small_classes_excluded(self, olivera_samples):
        """Classes with < 10 samples should be excluded."""
        counts = Counter(s["label"] for s in olivera_samples)
        for cls, n in counts.items():
            assert n >= 10, f"Class '{cls}' has only {n} samples"

    def test_unk_rate_under_one_percent(self, olivera_samples, vocab):
        """Olivera UNK rate should be negligible (~0.03%)."""
        unk_idx = vocab[cfg.UNK_TOKEN]
        total = sum(len(s["encoded"]) for s in olivera_samples)
        unk = sum(
            sum(1 for t in s["encoded"] if t == unk_idx)
            for s in olivera_samples
        )
        rate = unk / total if total > 0 else 0
        assert rate < 0.01, f"UNK rate {rate:.4f} exceeds 1%"

    def test_all_labels_in_encoder(self, olivera_samples, label_encoder):
        known = set(label_encoder.classes_)
        for s in olivera_samples:
            assert s["label"] in known


# ── Train/test split ────────────────────────────────────────────────────


class TestSplit:
    """Validate the train/test split properties."""

    def test_approximate_80_20_split(self, train_samples, test_samples):
        total = len(train_samples) + len(test_samples)
        test_ratio = len(test_samples) / total
        assert 0.18 < test_ratio < 0.22

    def test_stratified(self, train_samples, test_samples):
        """Each class should appear in both splits."""
        train_labels = set(s["label"] for s in train_samples)
        test_labels = set(s["label"] for s in test_samples)
        assert train_labels == test_labels
