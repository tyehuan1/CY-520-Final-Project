"""
V2 LSTM training entrypoint.

Differences vs the existing ``lstm_model.main()``:

* **Trains on the WITH-Trojan dataset** (``cfg.PREPROCESSED_TRAIN_PATH`` /
  ``cfg.PREPROCESSED_TEST_PATH`` / ``cfg.VOCABULARY_PATH``).
* **Restores the longer sequence-length sweep** ``[200, 500, 1000]``
  (``cfg.LSTM_V2_SEQUENCE_LENGTHS``) — the previous sweep
  ``[300, 400, 500]`` was a regression introduced to fit MalbehavD's
  short traces.
* **Writes to ``models/lstm_v2/``** so the existing ``models/lstm_no_trojan/``
  artifacts are untouched.
* The trained models are designed to be queried via
  :func:`src.model_training.lstm_model.predict_with_sliding_window` at
  evaluation time so that long sequences (up to ``MAX_RAW_SEQUENCE_LENGTH``)
  are scored end-to-end rather than truncated to the first window.

Usage::

    python -m src.model_training.train_lstm_v2
"""

import time

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import config as cfg
from src.data_loading.preprocessing import pad_sequences
from src.model_training.lstm_model import (
    build_lstm_model,
    predict_with_confidence,
    train_lstm,
)
from src.utils import get_logger, load_json, load_pickle, save_json, save_pickle

logger = get_logger(__name__)


def main() -> None:
    # ── Load with-Trojan data ────────────────────────────────────────────
    logger.info("Loading with-Trojan preprocessed data (8-class Mal-API)...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)
    vocab = load_json(cfg.VOCABULARY_PATH)

    classes = sorted({s["label"] for s in train_samples})
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    cfg.V2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_pickle(label_encoder, cfg.V2_LABEL_ENCODER_PATH)

    vocab_size = len(vocab)
    num_classes = len(label_encoder.classes_)

    train_encoded = [s["encoded"] for s in train_samples]
    test_encoded = [s["encoded"] for s in test_samples]

    y_train_int = label_encoder.transform([s["label"] for s in train_samples])
    y_test_int = label_encoder.transform([s["label"] for s in test_samples])
    y_train_onehot = to_categorical(y_train_int, num_classes=num_classes)

    logger.info(
        "Data loaded: %d train, %d test, vocab=%d, classes=%d (%s).",
        len(train_encoded), len(test_encoded), vocab_size, num_classes,
        list(label_encoder.classes_),
    )

    # ── Train at each V2 sequence length ────────────────────────────────
    results = {}

    for seq_len in cfg.LSTM_V2_SEQUENCE_LENGTHS:
        logger.info("=" * 70)
        logger.info("V2 TRAINING — SEQUENCE LENGTH = %d", seq_len)
        logger.info("=" * 70)

        X_train = pad_sequences(train_encoded, max_len=seq_len)
        X_test = pad_sequences(test_encoded, max_len=seq_len)
        logger.info(
            "Padded shapes: train=%s, test=%s.", X_train.shape, X_test.shape,
        )

        model = build_lstm_model(vocab_size, seq_len, num_classes)
        model.summary(print_fn=lambda x: logger.info(x))

        model_path = cfg.V2_LSTM_MODEL_DIR / f"best_len{seq_len}.keras"
        start = time.time()
        history = train_lstm(
            model, X_train, y_train_onehot, y_train_int, model_path,
        )
        elapsed = (time.time() - start) / 60.0
        logger.info(
            "Training time for seq_len=%d: %.1f minutes.", seq_len, elapsed,
        )

        train_preds, _ = predict_with_confidence(model, X_train)
        train_acc = float(np.mean(train_preds == y_train_int))
        train_macro_f1 = f1_score(y_train_int, train_preds, average="macro")

        preds, _ = predict_with_confidence(model, X_test)
        test_acc = float(np.mean(preds == y_test_int))
        test_macro_f1 = f1_score(y_test_int, preds, average="macro")

        report = classification_report(
            y_test_int, preds,
            target_names=label_encoder.classes_, output_dict=True,
        )
        report_str = classification_report(
            y_test_int, preds, target_names=label_encoder.classes_,
        )

        logger.info(
            "seq_len=%d — Train acc=%.4f F1=%.4f | Test acc=%.4f F1=%.4f",
            seq_len, train_acc, train_macro_f1, test_acc, test_macro_f1,
        )
        logger.info("\n%s", report_str)

        best_val_loss = min(history.history.get("val_loss", [float("inf")]))
        best_val_acc = max(history.history.get("val_accuracy", [0.0]))
        epochs_trained = len(history.history["loss"])

        results[seq_len] = {
            "seq_len": seq_len,
            "train_accuracy": round(train_acc, 4),
            "train_macro_f1": round(float(train_macro_f1), 4),
            "test_accuracy": round(test_acc, 4),
            "test_macro_f1": round(float(test_macro_f1), 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_accuracy": round(best_val_acc, 4),
            "epochs_trained": epochs_trained,
            "training_minutes": round(elapsed, 1),
            "with_trojan": True,
            "per_class_f1": {
                name: round(report[name]["f1-score"], 4)
                for name in label_encoder.classes_
            },
        }

        cfg.V2_METRICS_DIR.mkdir(parents=True, exist_ok=True)
        save_json(
            results[seq_len],
            cfg.V2_METRICS_DIR / f"lstm_v2_len{seq_len}_results.json",
        )

    # ── Summary ──────────────────────────────────────────────────────────
    best_len = max(results, key=lambda k: results[k]["test_macro_f1"])
    logger.info("=" * 70)
    logger.info(
        "V2 BEST SEQUENCE LENGTH: %d (test macro-F1 = %.4f)",
        best_len, results[best_len]["test_macro_f1"],
    )
    logger.info(
        "Update cfg.LSTM_V2_BEST_SEQ_LEN to %d before running V2 evaluation.",
        best_len,
    )
    logger.info("=" * 70)

    save_json(
        {
            "best_seq_len": best_len,
            "all_results": {str(k): v for k, v in results.items()},
        },
        cfg.V2_METRICS_DIR / "lstm_v2_comparison.json",
    )


if __name__ == "__main__":
    main()
