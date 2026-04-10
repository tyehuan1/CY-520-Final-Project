"""
V2 XGBoost training entrypoint.

Differences vs the existing ``xgboost_model.main()``:

* **Trains on the WITH-Trojan dataset** (``cfg.PREPROCESSED_TRAIN_PATH`` /
  ``cfg.PREPROCESSED_TEST_PATH``) — i.e. the original 8-class Mal-API
  preprocessed cache, not the no-Trojan filtered variant.
* **No augment_sequences()** — that helper was added to compensate for
  short MalbehavD-V1 sequences and produced training rows in the 20–200
  token range; the V2 model is meant to learn from full-length traces.
* **Length-normalized features with log-dampening** — calls
  ``build_feature_matrix(..., log_dampen=True)`` so that high-frequency
  CAPE tokens (e.g. 600k closehandle calls) can't dominate the
  category/bigram proportion vectors.
* **Writes to a separate ``models/xgboost_v2/`` directory** so the
  existing ``models/xgboost_no_trojan/`` artifacts are untouched.

Usage::

    python -m src.model_training.train_xgboost_v2
"""

import time

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

import config as cfg
from src.model_training.feature_engineering import (
    build_feature_matrix,
    build_tfidf_vectorizer,
)
from src.model_training.xgboost_model import (
    predict_with_confidence,
    save_model,
    train_xgboost,
)
from src.utils import get_logger, load_pickle, save_json, save_pickle

logger = get_logger(__name__)


def main() -> None:
    # ── Load with-Trojan preprocessed data ───────────────────────────────
    logger.info("Loading with-Trojan preprocessed data (8-class Mal-API)...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)

    classes = sorted({s["label"] for s in train_samples})
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    logger.info("Classes: %s", list(label_encoder.classes_))

    y_train = label_encoder.transform([s["label"] for s in train_samples])
    y_test = label_encoder.transform([s["label"] for s in test_samples])

    logger.info(
        "Dataset: %d train, %d test, %d classes.",
        len(train_samples), len(test_samples), len(label_encoder.classes_),
    )

    # ── Features (log-dampened) ──────────────────────────────────────────
    cfg.V2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.V2_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Building TF vectorizer on training data...")
    tfidf_vec = build_tfidf_vectorizer(train_samples)

    logger.info("Building log-dampened training feature matrix...")
    X_train = build_feature_matrix(train_samples, tfidf_vec, log_dampen=True)
    logger.info("Building log-dampened test feature matrix...")
    X_test = build_feature_matrix(test_samples, tfidf_vec, log_dampen=True)

    save_pickle(tfidf_vec, cfg.V2_TFIDF_PATH)
    save_pickle(label_encoder, cfg.V2_LABEL_ENCODER_PATH)
    save_pickle(X_train, cfg.V2_FEATURES_DIR / "X_train_xgb.pkl")
    save_pickle(X_test, cfg.V2_FEATURES_DIR / "X_test_xgb.pkl")
    logger.info(
        "Feature matrix shape: train=%s, test=%s", X_train.shape, X_test.shape,
    )

    # ── Train (V2 param dist — shallower trees, stronger regularization) ─
    logger.info("Starting V2 XGBoost training...")
    start = time.time()
    model, best_params = train_xgboost(
        X_train, y_train, label_encoder,
        param_dist=cfg.XGBOOST_V2_PARAM_DIST,
    )
    elapsed = (time.time() - start) / 60.0
    logger.info("Training time: %.1f minutes.", elapsed)

    cfg.V2_XGBOOST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, cfg.V2_XGBOOST_MODEL_DIR / "best_model.pkl")
    save_pickle(best_params, cfg.V2_XGBOOST_MODEL_DIR / "best_params.pkl")

    # ── Evaluate ─────────────────────────────────────────────────────────
    preds, _ = predict_with_confidence(model, X_test)
    test_acc = float(np.mean(preds == y_test))
    test_macro_f1 = f1_score(y_test, preds, average="macro")

    train_preds, _ = predict_with_confidence(model, X_train)
    train_acc = float(np.mean(train_preds == y_train))
    train_macro_f1 = f1_score(y_train, train_preds, average="macro")

    report = classification_report(
        y_test, preds, target_names=label_encoder.classes_, output_dict=True,
    )
    report_str = classification_report(
        y_test, preds, target_names=label_encoder.classes_,
    )
    logger.info(
        "V2 XGBoost — Train acc=%.4f Train macroF1=%.4f | Test acc=%.4f Test macroF1=%.4f",
        train_acc, train_macro_f1, test_acc, test_macro_f1,
    )
    logger.info("\n%s", report_str)

    cfg.V2_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "best_params": best_params,
            "train_accuracy": round(train_acc, 4),
            "train_macro_f1": round(float(train_macro_f1), 4),
            "test_accuracy": round(test_acc, 4),
            "test_macro_f1": round(float(test_macro_f1), 4),
            "training_minutes": round(elapsed, 1),
            "feature_matrix_shape": list(X_train.shape),
            "log_dampen": True,
            "with_trojan": True,
            "augment_sequences": False,
            "per_class_f1": {
                name: round(report[name]["f1-score"], 4)
                for name in label_encoder.classes_
            },
        },
        cfg.V2_METRICS_DIR / "xgboost_v2_results.json",
    )
    logger.info("V2 XGBoost results saved.")


if __name__ == "__main__":
    main()
