"""
End-to-end training script for both XGBoost and LSTM models.

Run from project root:
    python train_models.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import config as cfg
from src.data_loader import load_mal_api
from src.feature_engineering import build_feature_matrix, build_tfidf_vectorizer
from src.lstm_model import build_lstm_model, train_lstm
from src.lstm_model import predict_with_confidence as lstm_predict
from src.lstm_model import save_model as lstm_save
from src.preprocessing import (
    build_vocabulary,
    clean_samples,
    encode_samples,
    pad_sequences,
    stratified_split,
)
from src.utils import get_logger, save_pickle
from src.xgboost_model import train_xgboost
from src.xgboost_model import predict_with_confidence as xgb_predict
from src.xgboost_model import save_model as xgb_save

logger = get_logger(__name__)


def main():
    start = time.time()

    # ── 1. Load and preprocess ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preprocessing data")
    logger.info("=" * 60)

    samples = load_mal_api()
    cleaned = clean_samples(samples)
    train_samples, test_samples = stratified_split(cleaned)

    # Label encoding
    le = LabelEncoder()
    le.fit(cfg.MALWARE_FAMILIES)
    y_train = le.transform([s["label"] for s in train_samples])
    y_test = le.transform([s["label"] for s in test_samples])

    # Save label encoder
    save_pickle(le, cfg.CACHE_DIR / "label_encoder.pkl")

    # Vocabulary
    vocab = build_vocabulary(train_samples)
    from src.preprocessing import save_vocabulary
    save_vocabulary(vocab)

    # Encode for LSTM
    train_encoded = encode_samples(train_samples, vocab)
    test_encoded = encode_samples(test_samples, vocab)

    # Save preprocessed data
    save_pickle(train_encoded, cfg.PREPROCESSED_TRAIN_PATH)
    save_pickle(test_encoded, cfg.PREPROCESSED_TEST_PATH)

    logger.info(
        "Preprocessing done: %d train, %d test, vocab=%d.",
        len(train_samples), len(test_samples), len(vocab),
    )

    # ── 2. XGBoost training ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Training XGBoost")
    logger.info("=" * 60)

    tfidf_vec = build_tfidf_vectorizer(train_samples)
    save_pickle(tfidf_vec, cfg.CACHE_DIR / "tfidf_vectorizer.pkl")

    X_train_xgb = build_feature_matrix(train_samples, tfidf_vec)
    X_test_xgb = build_feature_matrix(test_samples, tfidf_vec)

    save_pickle(X_train_xgb, cfg.FEATURES_DIR / "X_train_xgb.pkl")
    save_pickle(X_test_xgb, cfg.FEATURES_DIR / "X_test_xgb.pkl")

    xgb_start = time.time()
    xgb_model, xgb_params = train_xgboost(X_train_xgb, y_train, le)
    xgb_time = time.time() - xgb_start

    xgb_save(xgb_model, cfg.XGBOOST_MODEL_DIR / "best_model.pkl")
    save_pickle(xgb_params, cfg.XGBOOST_MODEL_DIR / "best_params.pkl")

    # Quick eval on test set
    xgb_preds, xgb_probs = xgb_predict(xgb_model, X_test_xgb)
    xgb_acc = np.mean(xgb_preds == y_test)
    logger.info("XGBoost test accuracy: %.4f (trained in %.1fs)", xgb_acc, xgb_time)

    # ── 3. LSTM training (3 sequence lengths) ──────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Training LSTM (sequence lengths: %s)", cfg.LSTM_SEQUENCE_LENGTHS)
    logger.info("=" * 60)

    y_train_onehot = to_categorical(y_train, num_classes=cfg.NUM_CLASSES)
    vocab_size = len(vocab)

    best_lstm_f1 = -1.0
    best_lstm_seq_len = None
    lstm_results = {}

    for seq_len in cfg.LSTM_SEQUENCE_LENGTHS:
        logger.info("-" * 40)
        logger.info("Training LSTM with max_seq_len=%d", seq_len)
        logger.info("-" * 40)

        # Pad sequences
        X_train_lstm = pad_sequences(
            [s["encoded"] for s in train_encoded], max_len=seq_len
        )
        X_test_lstm = pad_sequences(
            [s["encoded"] for s in test_encoded], max_len=seq_len
        )

        model_path = cfg.LSTM_MODEL_DIR / f"lstm_seqlen{seq_len}.keras"

        lstm_start = time.time()
        model = build_lstm_model(vocab_size, seq_len, cfg.NUM_CLASSES)
        history = train_lstm(
            model, X_train_lstm, y_train_onehot, y_train, model_path
        )
        lstm_time = time.time() - lstm_start

        # Evaluate
        preds, probs = lstm_predict(model, X_test_lstm)
        acc = np.mean(preds == y_test)

        # Compute macro F1 for model selection
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, preds, average="macro")

        lstm_results[seq_len] = {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "train_time_s": lstm_time,
            "epochs_trained": len(history.history["loss"]),
        }

        logger.info(
            "LSTM seq_len=%d: accuracy=%.4f, macro_f1=%.4f, time=%.1fs, epochs=%d",
            seq_len, acc, f1, lstm_time, len(history.history["loss"]),
        )

        if f1 > best_lstm_f1:
            best_lstm_f1 = f1
            best_lstm_seq_len = seq_len
            lstm_save(model, cfg.LSTM_MODEL_DIR / "best_model.keras")

        # Save history
        hist_path = cfg.LSTM_MODEL_DIR / f"history_seqlen{seq_len}.json"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_path, "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # ── 4. Summary ─────────────────────────────────────────────────────
    total_time = time.time() - start

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("XGBoost: test accuracy=%.4f, time=%.1fs", xgb_acc, xgb_time)
    for sl, res in lstm_results.items():
        logger.info(
            "LSTM seq_len=%d: accuracy=%.4f, macro_f1=%.4f", sl, res["accuracy"], res["macro_f1"]
        )
    logger.info("Best LSTM: seq_len=%d (macro_f1=%.4f)", best_lstm_seq_len, best_lstm_f1)
    logger.info("Total time: %.1f minutes", total_time / 60)

    # Save summary
    summary = {
        "xgboost": {
            "test_accuracy": float(xgb_acc),
            "best_params": xgb_params,
            "train_time_s": xgb_time,
        },
        "lstm": lstm_results,
        "best_lstm_seq_len": best_lstm_seq_len,
        "total_time_s": total_time,
    }
    summary_path = cfg.RESULTS_DIR / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
