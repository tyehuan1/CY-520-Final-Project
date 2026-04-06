"""
Tests for Phase 6 & 7 evaluation scripts.

Validates:
  - metrics.py: compute_all_metrics correctness, plot functions don't crash
  - evaluate_models.py: artifact paths exist, feature name alignment
  - evaluate_generalizability.py: Olivera preprocessing helper, end-to-end
    routing logic, 8-class label construction, dual-preprocessing consistency
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

import config as cfg
from src.evaluation.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_predictions():
    """Synthetic binary classification results (2 classes)."""
    rng = np.random.RandomState(42)
    n = 100
    y_true = rng.randint(0, 2, size=n)
    y_prob = rng.dirichlet([1, 1], size=n)
    y_pred = np.argmax(y_prob, axis=1)
    class_names = ["Benign", "Malware"]
    return y_true, y_pred, y_prob, class_names


@pytest.fixture
def multiclass_predictions():
    """Synthetic 7-class classification results."""
    rng = np.random.RandomState(42)
    n = 200
    num_classes = 7
    y_true = rng.randint(0, num_classes, size=n)
    y_prob = rng.dirichlet(np.ones(num_classes), size=n)
    y_pred = np.argmax(y_prob, axis=1)
    class_names = list(cfg.MALWARE_FAMILIES)
    return y_true, y_pred, y_prob, class_names


# ---------------------------------------------------------------------------
# compute_all_metrics tests
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    """Tests for the core metrics computation function."""

    def test_returns_required_keys(self, binary_predictions):
        y_true, y_pred, y_prob, class_names = binary_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
        required = [
            "accuracy", "macro_f1", "weighted_f1",
            "roc_auc_macro", "roc_auc_weighted",
            "per_class", "confusion_matrix",
        ]
        for key in required:
            assert key in metrics, f"Missing key: {key}"

    def test_accuracy_in_range(self, multiclass_predictions):
        y_true, y_pred, y_prob, class_names = multiclass_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_f1_in_range(self, multiclass_predictions):
        y_true, y_pred, y_prob, class_names = multiclass_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["weighted_f1"] <= 1.0

    def test_per_class_has_all_families(self, multiclass_predictions):
        y_true, y_pred, y_prob, class_names = multiclass_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
        for name in class_names:
            assert name in metrics["per_class"]
            pc = metrics["per_class"][name]
            assert "precision" in pc
            assert "recall" in pc
            assert "f1" in pc
            assert "support" in pc

    def test_confusion_matrix_shape(self, multiclass_predictions):
        y_true, y_pred, y_prob, class_names = multiclass_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
        cm = metrics["confusion_matrix"]
        assert len(cm) == len(class_names)
        assert len(cm[0]) == len(class_names)

    def test_perfect_predictions(self):
        """Perfect predictions should yield accuracy=1.0 and macro-F1=1.0."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_all_metrics(y_true, y_pred, y_prob, ["A", "B", "C"])
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_roc_auc_binary(self, binary_predictions):
        """Binary ROC-AUC: label_binarize with 2 classes produces 1 column,
        so the multiclass OVR path in compute_all_metrics returns None.
        This is expected — binary ROC-AUC is handled separately in
        evaluate_models.py via sklearn.metrics.roc_auc_score directly."""
        y_true, y_pred, y_prob, class_names = binary_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
        # For 2-class, label_binarize returns shape (n, 1) which breaks
        # the multiclass OVR code path — None is the correct result here
        assert metrics["roc_auc_macro"] is None


# ---------------------------------------------------------------------------
# Artifact existence tests (evaluate_models.py dependencies)
# ---------------------------------------------------------------------------

class TestEvaluateModelsArtifacts:
    """Verify that all artifacts needed by evaluate_models.py exist."""

    def test_binary_model_exists(self):
        assert (cfg.BINARY_XGBOOST_MODEL_DIR / "best_model.pkl").exists()

    def test_binary_tfidf_exists(self):
        assert (cfg.BINARY_CACHE_DIR / "tfidf_vectorizer.pkl").exists()

    def test_binary_label_encoder_exists(self):
        assert cfg.BINARY_LABEL_ENCODER_PATH.exists()

    def test_binary_preprocessed_data_exists(self):
        assert cfg.BINARY_PREPROCESSED_TRAIN_PATH.exists()
        assert cfg.BINARY_PREPROCESSED_TEST_PATH.exists()

    def test_no_trojan_model_exists(self):
        assert (cfg.NO_TROJAN_XGBOOST_MODEL_DIR / "best_model.pkl").exists()

    def test_no_trojan_tfidf_exists(self):
        assert (cfg.NO_TROJAN_CACHE_DIR / "tfidf_vectorizer.pkl").exists()

    def test_no_trojan_label_encoder_exists(self):
        assert cfg.NO_TROJAN_LABEL_ENCODER_PATH.exists()

    def test_no_trojan_data_exists(self):
        assert cfg.NO_TROJAN_TRAIN_PATH.exists()
        assert cfg.NO_TROJAN_TEST_PATH.exists()

    def test_no_trojan_vocabulary_exists(self):
        assert cfg.NO_TROJAN_VOCABULARY_PATH.exists()

    def test_no_trojan_feature_caches_exist(self):
        assert (cfg.NO_TROJAN_FEATURES_DIR / "X_train_xgb.pkl").exists()
        assert (cfg.NO_TROJAN_FEATURES_DIR / "X_test_xgb.pkl").exists()

    def test_lstm_model_exists(self):
        assert (
            cfg.NO_TROJAN_LSTM_MODEL_DIR / f"best_len{cfg.LSTM_BEST_SEQ_LEN}.keras"
        ).exists()

    def test_ensemble_model_exists(self):
        assert (cfg.NO_TROJAN_ENSEMBLE_MODEL_DIR / "ensemble_model.pkl").exists()


# ---------------------------------------------------------------------------
# Artifact existence tests (evaluate_generalizability.py dependencies)
# ---------------------------------------------------------------------------

class TestGeneralizabilityArtifacts:
    """Verify all artifacts needed by evaluate_generalizability.py exist."""

    def test_no_trojan_malbehavd_exists(self):
        assert cfg.NO_TROJAN_MALBEHAVD_PATH.exists()

    def test_binary_artifacts_exist(self):
        """Stage-1 binary model artifacts must be present."""
        assert (cfg.BINARY_XGBOOST_MODEL_DIR / "best_model.pkl").exists()
        assert (cfg.BINARY_CACHE_DIR / "tfidf_vectorizer.pkl").exists()
        assert cfg.BINARY_LABEL_ENCODER_PATH.exists()


# ---------------------------------------------------------------------------
# Olivera preprocessing helper tests
# ---------------------------------------------------------------------------

class TestOliveraPreprocessHelper:
    """Tests for _olivera_preprocess_malbehavd in evaluate_generalizability."""

    def test_olivera_preprocess_deduplicates(self):
        from src.evaluation.evaluate_generalizability import _olivera_preprocess_malbehavd

        samples = [{
            "sequence": ["NtOpenFile", "NtOpenFile", "NtOpenFile", "NtReadFile"],
            "label": "Backdoor",
        }]
        result = _olivera_preprocess_malbehavd(samples)
        # After lowercase + Olivera dedup: ntopenfile, ntreadfile
        assert result[0]["sequence"] == ["ntopenfile", "ntreadfile"]

    def test_olivera_preprocess_truncates(self):
        from src.evaluation.evaluate_generalizability import _olivera_preprocess_malbehavd

        # 200 unique calls should be truncated to 100
        long_seq = [f"api_{i}" for i in range(200)]
        samples = [{"sequence": long_seq, "label": "Virus"}]
        result = _olivera_preprocess_malbehavd(samples)
        assert len(result[0]["sequence"]) == 100

    def test_olivera_preprocess_preserves_label(self):
        from src.evaluation.evaluate_generalizability import _olivera_preprocess_malbehavd

        samples = [{"sequence": ["NtClose"], "label": "Spyware", "sha256": "abc"}]
        result = _olivera_preprocess_malbehavd(samples)
        assert result[0]["label"] == "Spyware"
        assert result[0]["sha256"] == "abc"


# ---------------------------------------------------------------------------
# End-to-end routing logic tests
# ---------------------------------------------------------------------------

class TestEndToEndRouting:
    """Tests for the Stage-1 → Stage-2 routing logic."""

    def test_benign_routing_at_50_pct(self):
        """Samples with P(malware) < 0.50 should be labeled Benign."""
        from src.evaluation.evaluate_generalizability import BINARY_THRESHOLD

        # Simulate p_malware for 5 samples
        p_malware = np.array([0.10, 0.49, 0.50, 0.80, 0.99])
        passes = (p_malware >= BINARY_THRESHOLD).astype(int)

        assert passes[0] == 0  # 0.10 → Benign
        assert passes[1] == 0  # 0.49 → Benign
        assert passes[2] == 1  # 0.50 → Stage-2
        assert passes[3] == 1  # 0.80 → Stage-2
        assert passes[4] == 1  # 0.99 → Stage-2

    def test_e2e_class_names_are_8(self):
        """End-to-end evaluation should use 8 classes: Benign + 7 families."""
        e2e_class_names = [cfg.BENIGN_LABEL] + list(cfg.MALWARE_FAMILIES)
        assert len(e2e_class_names) == 8
        assert e2e_class_names[0] == "Benign"
        assert "Trojan" not in e2e_class_names

    def test_blocked_samples_labeled_benign(self):
        """Samples blocked by Stage-1 must get 'Benign' label regardless of
        what Stage-2 would have predicted."""
        passes_stage1 = np.array([0, 1, 0, 1, 0])  # 0=blocked, 1=passed
        family_pred_labels = np.array(["Virus", "Backdoor", "Spyware", "Worms", "Adware"])

        pred_labels = []
        for i in range(5):
            if passes_stage1[i]:
                pred_labels.append(family_pred_labels[i])
            else:
                pred_labels.append(cfg.BENIGN_LABEL)

        assert pred_labels == ["Benign", "Backdoor", "Benign", "Worms", "Benign"]

    def test_all_benign_leaked_get_family_label(self):
        """Benign samples that pass Stage-1 will always get a family label
        (Stage-2 has no Benign class), which is always a misclassification."""
        # Simulate: 3 benign samples all pass Stage-1
        true_labels = ["Benign", "Benign", "Benign"]
        passes_stage1 = np.array([1, 1, 1])
        family_preds = ["Backdoor", "Virus", "Spyware"]

        pred_labels = []
        for i in range(3):
            if passes_stage1[i]:
                pred_labels.append(family_preds[i])
            else:
                pred_labels.append(cfg.BENIGN_LABEL)

        # All predictions are wrong — benign samples got family labels
        for true, pred in zip(true_labels, pred_labels):
            assert true != pred  # Every prediction is a misclassification


# ---------------------------------------------------------------------------
# Feature name alignment tests
# ---------------------------------------------------------------------------

class TestFeatureNameAlignment:
    """Verify feature name lists match trained model dimensions."""

    def test_binary_feature_names_match_model(self):
        from src.model_training.binary_xgboost_model import BINARY_STATISTICAL_FEATURE_NAMES
        from src.model_training.feature_engineering import CATEGORY_FEATURE_NAMES
        from src.utils import load_pickle

        bin_tfidf = load_pickle(cfg.BINARY_CACHE_DIR / "tfidf_vectorizer.pkl")
        X_test = load_pickle(cfg.BINARY_FEATURES_DIR / "X_test_binary_xgb.pkl")

        feature_names = (
            bin_tfidf.get_feature_names_out().tolist()
            + list(BINARY_STATISTICAL_FEATURE_NAMES)
            + CATEGORY_FEATURE_NAMES
        )
        assert len(feature_names) == X_test.shape[1], (
            f"Feature names ({len(feature_names)}) != matrix cols ({X_test.shape[1]})"
        )

    def test_family_feature_names_match_model(self):
        from src.model_training.feature_engineering import (
            CATEGORY_FEATURE_NAMES,
            STATISTICAL_FEATURE_NAMES,
        )
        from src.utils import load_pickle

        tfidf = load_pickle(cfg.NO_TROJAN_CACHE_DIR / "tfidf_vectorizer.pkl")
        X_test = load_pickle(cfg.NO_TROJAN_FEATURES_DIR / "X_test_xgb.pkl")

        feature_names = (
            tfidf.get_feature_names_out().tolist()
            + list(STATISTICAL_FEATURE_NAMES)
            + CATEGORY_FEATURE_NAMES
        )
        assert len(feature_names) == X_test.shape[1], (
            f"Feature names ({len(feature_names)}) != matrix cols ({X_test.shape[1]})"
        )


# ---------------------------------------------------------------------------
# No-Trojan label encoder consistency
# ---------------------------------------------------------------------------

class TestNoTrojanConsistency:
    """Verify Trojan is excluded from the no-Trojan pipeline."""

    def test_label_encoder_has_7_classes(self):
        from src.utils import load_pickle
        le = load_pickle(cfg.NO_TROJAN_LABEL_ENCODER_PATH)
        assert len(le.classes_) == 7
        assert "Trojan" not in le.classes_

    def test_config_families_match_encoder(self):
        from src.utils import load_pickle
        le = load_pickle(cfg.NO_TROJAN_LABEL_ENCODER_PATH)
        assert set(le.classes_) == set(cfg.MALWARE_FAMILIES)

    def test_malbehavd_no_trojan_has_no_trojan(self):
        from src.utils import load_json
        data = load_json(cfg.NO_TROJAN_MALBEHAVD_PATH)
        labels = {s["label"] for s in data["samples"]}
        assert "Trojan" not in labels
