"""
Tests for src.preprocessing — cleaning, vocabulary, encoding, splitting.
"""

from collections import Counter

import numpy as np
import pytest

import config as cfg
from src.data_loading.data_loader import load_mal_api
from src.data_loading.preprocessing import (
    build_vocabulary,
    clean_samples,
    clean_sequence,
    collapse_consecutive_duplicates,
    compute_unk_ratio,
    encode_samples,
    encode_sequence,
    pad_sequences,
    remove_sandbox_tokens,
    stratified_split,
    winmet_to_mal_api_format,
    winmet_to_mal_api_format_no_trojan,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def raw_samples():
    """Load raw Mal-API-2019 samples once."""
    return load_mal_api()


@pytest.fixture(scope="module")
def cleaned_samples(raw_samples):
    """Clean all raw samples once."""
    return clean_samples(raw_samples)


@pytest.fixture(scope="module")
def split_data(cleaned_samples):
    """Stratified split once for the module."""
    return stratified_split(cleaned_samples)


@pytest.fixture(scope="module")
def train_samples(split_data):
    return split_data[0]


@pytest.fixture(scope="module")
def test_samples(split_data):
    return split_data[1]


@pytest.fixture(scope="module")
def vocab(train_samples):
    return build_vocabulary(train_samples)


# ── Sandbox token removal ─────────────────────────────────────────────────


class TestRemoveSandboxTokens:
    """Tests for __exception__, __anomaly__, etc. removal."""

    def test_exception_removed(self):
        seq = ["ntclose", "__exception__", "ntopen"]
        assert remove_sandbox_tokens(seq) == ["ntclose", "ntopen"]

    def test_anomaly_removed(self):
        seq = ["ntclose", "__anomaly__", "ntopen"]
        assert remove_sandbox_tokens(seq) == ["ntclose", "ntopen"]

    def test_multiple_dunder_removed(self):
        seq = ["__exception__", "a", "__anomaly__", "b", "__other__"]
        assert remove_sandbox_tokens(seq) == ["a", "b"]

    def test_no_dunder_unchanged(self):
        seq = ["ntclose", "ntopen", "ldrloaddll"]
        assert remove_sandbox_tokens(seq) == seq

    def test_empty_sequence(self):
        assert remove_sandbox_tokens([]) == []

    def test_all_samples_cleaned(self, cleaned_samples):
        """No cleaned sample should contain tokens starting with __."""
        for s in cleaned_samples:
            for tok in s["sequence"]:
                assert not tok.startswith("__"), f"Found dunder token: {tok}"


# ── Consecutive duplicate collapsing ──────────────────────────────────────


class TestCollapseDuplicates:
    """Tests for collapsing consecutive repeated API calls."""

    def test_long_run_truncated(self):
        seq = ["a"] * 10
        result = collapse_consecutive_duplicates(seq, max_repeats=5)
        assert result == ["a"] * 5

    def test_short_run_preserved(self):
        seq = ["a", "a", "a"]
        result = collapse_consecutive_duplicates(seq, max_repeats=5)
        assert result == ["a", "a", "a"]

    def test_exact_limit_preserved(self):
        seq = ["a"] * 5
        result = collapse_consecutive_duplicates(seq, max_repeats=5)
        assert result == ["a"] * 5

    def test_mixed_runs(self):
        seq = ["a"] * 7 + ["b"] * 3 + ["a"] * 8
        result = collapse_consecutive_duplicates(seq, max_repeats=5)
        assert result == ["a"] * 5 + ["b"] * 3 + ["a"] * 5

    def test_alternating_not_collapsed(self):
        seq = ["a", "b", "a", "b", "a", "b", "a", "b"]
        result = collapse_consecutive_duplicates(seq, max_repeats=5)
        assert result == seq

    def test_empty_sequence(self):
        assert collapse_consecutive_duplicates([], max_repeats=5) == []

    def test_single_token(self):
        assert collapse_consecutive_duplicates(["a"], max_repeats=5) == ["a"]

    def test_hand_crafted_realistic(self):
        """Realistic API sequence with a long ldrgetprocedureaddress run."""
        seq = (
            ["ldrloaddll"]
            + ["ldrgetprocedureaddress"] * 20
            + ["ntclose"]
        )
        result = collapse_consecutive_duplicates(seq, max_repeats=5)
        expected = (
            ["ldrloaddll"]
            + ["ldrgetprocedureaddress"] * 5
            + ["ntclose"]
        )
        assert result == expected


# ── Vocabulary ─────────────────────────────────────────────────────────────


class TestVocabulary:
    """Tests for vocabulary construction."""

    def test_pad_at_index_zero(self, vocab):
        assert vocab[cfg.PAD_TOKEN] == cfg.PAD_INDEX

    def test_unk_at_index_one(self, vocab):
        assert vocab[cfg.UNK_TOKEN] == cfg.UNK_INDEX

    def test_all_training_tokens_in_vocab(self, train_samples, vocab):
        """Every token in the training set must appear in the vocabulary."""
        for s in train_samples:
            for tok in s["sequence"]:
                assert tok in vocab, f"Training token '{tok}' missing from vocab"

    def test_vocab_indices_are_contiguous(self, vocab):
        """Indices should be 0, 1, 2, ..., len(vocab)-1."""
        indices = sorted(vocab.values())
        assert indices == list(range(len(vocab)))

    def test_vocab_size_reasonable(self, vocab):
        """Vocabulary should be in the ballpark of ~280 unique API calls + 2."""
        assert 100 < len(vocab) < 500


# ── Encoding ───────────────────────────────────────────────────────────────


class TestEncoding:
    """Tests for sequence-to-integer encoding."""

    def test_known_tokens_encoded(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "ntclose": 2, "ntopen": 3}
        result = encode_sequence(["ntclose", "ntopen", "ntclose"], vocab)
        assert result == [2, 3, 2]

    def test_unknown_maps_to_unk(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "ntclose": 2}
        result = encode_sequence(["ntclose", "unknownapi"], vocab)
        assert result == [2, 1]

    def test_all_encoded_indices_valid(self, train_samples, test_samples, vocab):
        """Every encoded index must be within [0, vocab_size)."""
        max_idx = len(vocab) - 1
        for samples in (train_samples, test_samples):
            encoded = encode_samples(samples, vocab)
            for s in encoded:
                for idx in s["encoded"]:
                    assert 0 <= idx <= max_idx, f"Index {idx} out of range"

    def test_encoded_length_matches_sequence(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3}
        seq = ["a", "b", "a"]
        result = encode_sequence(seq, vocab)
        assert len(result) == len(seq)


# ── Padding ────────────────────────────────────────────────────────────────


class TestPadSequences:
    """Tests for pad/truncate to fixed length."""

    def test_short_sequence_padded(self):
        seqs = [[1, 2, 3]]
        result = pad_sequences(seqs, max_len=5)
        expected = np.array([[1, 2, 3, 0, 0]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_long_sequence_truncated(self):
        seqs = [[1, 2, 3, 4, 5, 6, 7]]
        result = pad_sequences(seqs, max_len=4)
        expected = np.array([[1, 2, 3, 4]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_exact_length_unchanged(self):
        seqs = [[1, 2, 3]]
        result = pad_sequences(seqs, max_len=3)
        expected = np.array([[1, 2, 3]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_output_shape(self):
        seqs = [[1, 2], [3, 4, 5, 6]]
        result = pad_sequences(seqs, max_len=10)
        assert result.shape == (2, 10)


# ── Stratified split ──────────────────────────────────────────────────────


class TestStratifiedSplit:
    """Tests for the train/test split."""

    def test_split_sizes(self, train_samples, test_samples):
        """80/20 split of 7107 samples."""
        total = len(train_samples) + len(test_samples)
        assert total == cfg.MAL_API_EXPECTED_SAMPLES
        assert abs(len(test_samples) / total - cfg.TEST_SIZE) < 0.01

    def test_class_ratios_preserved(self, raw_samples, train_samples, test_samples):
        """Per-class ratios in train and test should be within ±2% of original."""
        orig_counts = Counter(s["label"] for s in raw_samples)
        train_counts = Counter(s["label"] for s in train_samples)
        test_counts = Counter(s["label"] for s in test_samples)

        # Use the full 8-family set since raw_samples includes Trojan
        raw_families = [
            "Adware", "Backdoor", "Downloader", "Dropper",
            "Spyware", "Trojan", "Virus", "Worms",
        ]
        for label in raw_families:
            orig_ratio = orig_counts[label] / len(raw_samples)
            train_ratio = train_counts[label] / len(train_samples)
            test_ratio = test_counts[label] / len(test_samples)

            assert abs(train_ratio - orig_ratio) < 0.02, (
                f"Train ratio for {label}: {train_ratio:.3f} vs {orig_ratio:.3f}"
            )
            assert abs(test_ratio - orig_ratio) < 0.02, (
                f"Test ratio for {label}: {test_ratio:.3f} vs {orig_ratio:.3f}"
            )

    def test_no_data_leakage(self, train_samples, test_samples):
        """No sample object should appear in both train and test sets.

        We compare by object identity (id) since the dataset may contain
        legitimately duplicated sequences across different malware samples.
        """
        train_ids = {id(s) for s in train_samples}
        test_ids = {id(s) for s in test_samples}
        overlap = train_ids & test_ids
        assert len(overlap) == 0, f"Found {len(overlap)} shared sample objects"

        # Also verify total count is preserved (no samples lost or doubled)
        assert len(train_samples) + len(test_samples) == cfg.MAL_API_EXPECTED_SAMPLES

    def test_deterministic_with_seed(self, cleaned_samples):
        """Same seed should produce identical splits."""
        t1, s1 = stratified_split(cleaned_samples)
        t2, s2 = stratified_split(cleaned_samples)
        assert len(t1) == len(t2)
        # Check first 10 labels match
        for a, b in zip(t1[:10], t2[:10]):
            assert a["label"] == b["label"]
            assert a["sequence"] == b["sequence"]


# ── UNK ratio ──────────────────────────────────────────────────────────────


class TestUnkRatio:
    """Tests for unknown-token ratio computation."""

    def test_all_known(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3}
        samples = [{"sequence": ["a", "b", "a"]}]
        assert compute_unk_ratio(samples, vocab) == 0.0

    def test_all_unknown(self):
        vocab = {"<PAD>": 0, "<UNK>": 1}
        samples = [{"sequence": ["x", "y"]}]
        assert compute_unk_ratio(samples, vocab) == 1.0

    def test_half_unknown(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2}
        samples = [{"sequence": ["a", "x"]}]
        assert compute_unk_ratio(samples, vocab) == 0.5

    def test_empty_samples(self):
        vocab = {"<PAD>": 0, "<UNK>": 1}
        samples = [{"sequence": []}]
        assert compute_unk_ratio(samples, vocab) == 0.0


# ── WinMET → Mal-API format conversion ────────────────────────────────────


class TestWinmetToMalApiFormat:
    """Tests for winmet_to_mal_api_format and its no-Trojan variant.

    Builds a tiny synthetic WinMET-style Parquet so the tests don't depend
    on the multi-GB real extraction.
    """

    @pytest.fixture
    def fake_winmet_parquet(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [
            ("hash1", "redline",   "Spyware",    "ntopenfile ntclose"),
            ("hash2", "berbew",    "Spyware",    "ntcreatefile sysenter"),
            ("hash3", "amadey",    "Downloader", "regopenkey ntclose"),
            ("hash4", "djvu",      "Trojan",     "ntopenfile ntclose ntopenfile"),
            ("hash5", "stop",      "Trojan",     "regopenkey regclose"),
            ("hash6", "vbclone",   "Worms",      "ntcreatefile ntclose"),
            ("hash7", "unknownfam", None,        "ntclose"),  # dropped (no class)
            ("hash8", "unknownfam", "",          "ntclose"),  # dropped (empty class)
        ]
        table = pa.table({
            "sha256":           pa.array([r[0] for r in rows], type=pa.string()),
            "family_avclass":   pa.array([r[1] for r in rows], type=pa.string()),
            "family_cape":      pa.array(["x"] * len(rows), type=pa.string()),
            "family_consensus": pa.array(["x"] * len(rows), type=pa.string()),
            "primary_class":    pa.array([r[2] for r in rows], type=pa.string()),
            "secondary_classes": pa.array([""] * len(rows), type=pa.string()),
            "api_sequence":     pa.array([r[3] for r in rows], type=pa.string()),
            "num_processes":    pa.array([1] * len(rows), type=pa.int32()),
            "sequence_length":  pa.array([len(r[3].split()) for r in rows], type=pa.int32()),
            "source_volume":    pa.array([1] * len(rows), type=pa.int32()),
        })
        path = tmp_path / "fake_winmet.parquet"
        pq.write_table(table, path, compression="snappy")
        return path

    def test_full_conversion_drops_unmapped(self, fake_winmet_parquet, tmp_path):
        seq_out = tmp_path / "seqs.txt"
        lbl_out = tmp_path / "lbls.csv"
        n = winmet_to_mal_api_format(
            parquet_path=fake_winmet_parquet,
            sequences_out=seq_out,
            labels_out=lbl_out,
        )
        # 8 rows total - 2 with null/empty primary_class = 6 written
        assert n == 6
        assert seq_out.exists() and lbl_out.exists()

    def test_full_conversion_keeps_trojan(self, fake_winmet_parquet, tmp_path):
        seq_out = tmp_path / "seqs.txt"
        lbl_out = tmp_path / "lbls.csv"
        winmet_to_mal_api_format(
            parquet_path=fake_winmet_parquet,
            sequences_out=seq_out,
            labels_out=lbl_out,
        )
        labels = lbl_out.read_text(encoding="utf-8").strip().splitlines()
        assert "Trojan" in labels
        assert labels.count("Trojan") == 2

    def test_no_trojan_variant_drops_trojan(self, fake_winmet_parquet, tmp_path):
        seq_out = tmp_path / "seqs.txt"
        lbl_out = tmp_path / "lbls.csv"
        n = winmet_to_mal_api_format_no_trojan(
            parquet_path=fake_winmet_parquet,
            sequences_out=seq_out,
            labels_out=lbl_out,
        )
        # 6 mapped rows - 2 Trojan = 4 written
        assert n == 4
        labels = lbl_out.read_text(encoding="utf-8").strip().splitlines()
        assert "Trojan" not in labels
        assert set(labels) == {"Spyware", "Downloader", "Worms"}

    def test_output_loadable_via_load_mal_api(self, fake_winmet_parquet, tmp_path):
        """Files must round-trip through load_mal_api unchanged."""
        seq_out = tmp_path / "seqs.txt"
        lbl_out = tmp_path / "lbls.csv"
        winmet_to_mal_api_format(
            parquet_path=fake_winmet_parquet,
            sequences_out=seq_out,
            labels_out=lbl_out,
        )
        samples = load_mal_api(
            sequences_path=seq_out,
            labels_path=lbl_out,
            max_seq_len=None,
        )
        assert len(samples) == 6
        # Sequences are space-tokenized lists of lowercase API call strings
        for s in samples:
            assert isinstance(s["sequence"], list)
            assert all(isinstance(t, str) for t in s["sequence"])
            assert all(t == t.lower() for t in s["sequence"])
            assert s["label"] in {"Spyware", "Downloader", "Worms", "Trojan"}

    def test_line_counts_match(self, fake_winmet_parquet, tmp_path):
        """Sequences file and labels file must have identical line counts."""
        seq_out = tmp_path / "seqs.txt"
        lbl_out = tmp_path / "lbls.csv"
        winmet_to_mal_api_format(
            parquet_path=fake_winmet_parquet,
            sequences_out=seq_out,
            labels_out=lbl_out,
        )
        n_seq = sum(1 for _ in open(seq_out, encoding="utf-8"))
        n_lbl = sum(1 for _ in open(lbl_out, encoding="utf-8"))
        assert n_seq == n_lbl == 6
