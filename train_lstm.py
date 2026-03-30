"""
Train Bidirectional LSTM at three sequence lengths (200, 500, 1000).

Loads cached preprocessed data, pads to each length, trains, evaluates on
test set, and selects the best model by validation macro-F1.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.utils import to_categorical

import config as cfg
from src.lstm_model import (
    build_lstm_model,
    predict_with_confidence,
    save_model,
    train_lstm,
)
from src.preprocessing import pad_sequences
from src.utils import get_logger, load_json, load_pickle, save_json

logger = get_logger(__name__)


def main() -> None:
    # ── Load cached data ─────────────────────────────────────────────────
    logger.info("Loading cached preprocessed data...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)
    vocab = load_json(cfg.VOCABULARY_PATH)
    label_encoder = load_pickle(cfg.CACHE_DIR / "label_encoder.pkl")

    vocab_size = len(vocab)
    num_classes = len(label_encoder.classes_)

    # Extract encoded sequences and labels
    train_encoded = [s["encoded"] for s in train_samples]
    test_encoded = [s["encoded"] for s in test_samples]

    y_train_int = label_encoder.transform([s["label"] for s in train_samples])
    y_test_int = label_encoder.transform([s["label"] for s in test_samples])

    y_train_onehot = to_categorical(y_train_int, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test_int, num_classes=num_classes)

    logger.info(
        "Data loaded: %d train, %d test, vocab=%d, classes=%d.",
        len(train_encoded), len(test_encoded), vocab_size, num_classes,
    )

    # ── Train at each sequence length ────────────────────────────────────
    results = {}

    for seq_len in cfg.LSTM_SEQUENCE_LENGTHS:
        logger.info("=" * 70)
        logger.info("TRAINING WITH SEQUENCE LENGTH = %d", seq_len)
        logger.info("=" * 70)

        # Pad sequences
        X_train = pad_sequences(train_encoded, max_len=seq_len)
        X_test = pad_sequences(test_encoded, max_len=seq_len)
        logger.info("Padded shapes: train=%s, test=%s.", X_train.shape, X_test.shape)

        # Build model
        model = build_lstm_model(vocab_size, seq_len, num_classes)
        model.summary(print_fn=lambda x: logger.info(x))

        # Train
        model_path = cfg.LSTM_MODEL_DIR / f"best_len{seq_len}.keras"
        start = time.time()
        history = train_lstm(
            model, X_train, y_train_onehot, y_train_int, model_path,
        )
        elapsed = (time.time() - start) / 60.0
        logger.info("Training time for seq_len=%d: %.1f minutes.", seq_len, elapsed)

        # Evaluate on test set
        preds, probs = predict_with_confidence(model, X_test)
        test_acc = np.mean(preds == y_test_int)
        test_macro_f1 = f1_score(y_test_int, preds, average="macro")

        report = classification_report(
            y_test_int, preds,
            target_names=label_encoder.classes_,
            output_dict=True,
        )
        report_str = classification_report(
            y_test_int, preds,
            target_names=label_encoder.classes_,
        )

        logger.info("seq_len=%d — Test accuracy: %.4f, Test macro-F1: %.4f",
                     seq_len, test_acc, test_macro_f1)
        logger.info("\n%s", report_str)

        # Best validation loss/accuracy from history
        best_val_loss = min(history.history.get("val_loss", [float("inf")]))
        best_val_acc = max(history.history.get("val_accuracy", [0.0]))
        epochs_trained = len(history.history["loss"])

        results[seq_len] = {
            "seq_len": seq_len,
            "test_accuracy": round(test_acc, 4),
            "test_macro_f1": round(test_macro_f1, 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_accuracy": round(best_val_acc, 4),
            "epochs_trained": epochs_trained,
            "training_minutes": round(elapsed, 1),
            "per_class_f1": {
                name: round(report[name]["f1-score"], 4)
                for name in label_encoder.classes_
            },
        }

        # Save per-length results
        cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        save_json(
            results[seq_len],
            cfg.METRICS_DIR / f"lstm_len{seq_len}_results.json",
        )

    # ── Select best model ────────────────────────────────────────────────
    best_len = max(results, key=lambda k: results[k]["test_macro_f1"])
    logger.info("=" * 70)
    logger.info("BEST SEQUENCE LENGTH: %d (macro-F1 = %.4f)",
                best_len, results[best_len]["test_macro_f1"])
    logger.info("=" * 70)

    # Save summary
    summary = {
        "best_seq_len": best_len,
        "all_results": {str(k): v for k, v in results.items()},
    }
    save_json(summary, cfg.METRICS_DIR / "lstm_comparison.json")
    logger.info("All LSTM results saved to %s.", cfg.METRICS_DIR / "lstm_comparison.json")


if __name__ == "__main__":
    main()
