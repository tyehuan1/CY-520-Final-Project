"""
Bidirectional LSTM model: architecture, training, saving, loading, inference,
and end-to-end training pipeline.

Uses Keras (TensorFlow backend) with the architecture:
Embedding → SpatialDropout1D → BiLSTM(128) → BiLSTM(64) → Dense(64) → Dense(N).

Run directly to train at multiple sequence lengths::

    python -m src.lstm_model
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Suppress TF warnings before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

import config as cfg
from src.data_loading.preprocessing import pad_sequences
from src.utils import get_logger, load_json, load_pickle, save_json

logger = get_logger(__name__)


def build_lstm_model(
    vocab_size: int,
    max_seq_len: int,
    num_classes: int = cfg.NUM_CLASSES,
) -> keras.Model:
    """Build and compile a Bidirectional LSTM classifier.

    Architecture:
        Embedding(128) → SpatialDropout1D(0.2) →
        BiLSTM(128, return_sequences=True, recurrent_dropout=0.1) →
        BiLSTM(64, recurrent_dropout=0.1) →
        Dense(64, relu) → Dropout(0.3) → Dense(num_classes, softmax)

    Args:
        vocab_size: Number of tokens in the vocabulary (including PAD/UNK).
        max_seq_len: Fixed input sequence length (after padding/truncation).
        num_classes: Number of output classes.

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential([
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=cfg.LSTM_EMBEDDING_DIM,
            mask_zero=True,
        ),
        layers.SpatialDropout1D(cfg.LSTM_SPATIAL_DROPOUT),
        layers.Bidirectional(
            layers.LSTM(
                cfg.LSTM_HIDDEN_UNITS_1,
                return_sequences=True,
                recurrent_dropout=cfg.LSTM_RECURRENT_DROPOUT,
            )
        ),
        layers.Bidirectional(
            layers.LSTM(
                cfg.LSTM_HIDDEN_UNITS_2,
                return_sequences=False,
                recurrent_dropout=cfg.LSTM_RECURRENT_DROPOUT,
            )
        ),
        layers.Dense(cfg.LSTM_DENSE_UNITS, activation="relu"),
        layers.Dropout(cfg.LSTM_DROPOUT),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.LSTM_INITIAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Build the model so we can log the summary
    model.build(input_shape=(None, max_seq_len))
    logger.info(
        "LSTM model built: vocab=%d, max_seq_len=%d, classes=%d, params=%s.",
        vocab_size,
        max_seq_len,
        num_classes,
        f"{model.count_params():,}",
    )
    return model


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalanced training data.

    Args:
        y_train: Integer-encoded labels.

    Returns:
        Dict mapping class index to weight.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes.tolist(), weights.tolist()))


def get_callbacks(
    model_path: Path,
    monitor: str = "val_loss",
) -> List[keras.callbacks.Callback]:
    """Create training callbacks: ReduceLROnPlateau, EarlyStopping, ModelCheckpoint.

    Args:
        model_path: Path to save the best model checkpoint.
        monitor: Metric to monitor for callbacks.

    Returns:
        List of Keras callbacks.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=cfg.LR_REDUCE_FACTOR,
            patience=cfg.LR_REDUCE_PATIENCE,
            min_lr=cfg.LR_MIN,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=cfg.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_lstm(
    model: keras.Model,
    X_train: np.ndarray,
    y_train_onehot: np.ndarray,
    y_train_int: np.ndarray,
    model_path: Path,
    batch_size: int = cfg.LSTM_BATCH_SIZE,
    max_epochs: int = cfg.LSTM_MAX_EPOCHS,
    validation_split: float = cfg.LSTM_VALIDATION_SPLIT,
) -> keras.callbacks.History:
    """Train the LSTM model with callbacks and class weights.

    Args:
        model: Compiled Keras model.
        X_train: Padded integer-encoded sequences, shape ``(n, max_len)``.
        y_train_onehot: One-hot encoded labels, shape ``(n, num_classes)``.
        y_train_int: Integer-encoded labels (for class weight computation).
        model_path: Path for the ModelCheckpoint callback.
        batch_size: Training batch size.
        max_epochs: Maximum number of epochs.
        validation_split: Fraction of training data for validation.

    Returns:
        Keras History object with training/validation metrics.
    """
    class_weights = compute_class_weights(y_train_int)
    callbacks = get_callbacks(model_path)

    logger.info(
        "Training LSTM: %d samples, batch_size=%d, max_epochs=%d.",
        len(X_train),
        batch_size,
        max_epochs,
    )

    history = model.fit(
        X_train,
        y_train_onehot,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_split=validation_split,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info(
        "Training complete. Final val_loss=%.4f, val_accuracy=%.4f.",
        min(history.history.get("val_loss", [float("inf")])),
        max(history.history.get("val_accuracy", [0.0])),
    )
    return history


def predict_with_confidence(
    model: keras.Model,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions and per-class probability scores.

    Args:
        model: Trained Keras model.
        X: Padded integer-encoded sequences, shape ``(n, max_len)``.

    Returns:
        Tuple of (predicted_labels, probabilities) where labels has shape
        ``(n,)`` and probabilities has shape ``(n, num_classes)``.
    """
    probabilities = model.predict(X, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities


def save_model(model: keras.Model, path: Path) -> None:
    """Save a Keras model to disk.

    Args:
        model: Trained model.
        path: Destination path (directory for SavedModel format).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.info("LSTM model saved to %s.", path)


def load_model(path: Path) -> keras.Model:
    """Load a Keras model from disk.

    Args:
        path: Path to saved model.

    Returns:
        Loaded Keras model.
    """
    model = keras.models.load_model(str(path))
    logger.info("LSTM model loaded from %s.", path)
    return model


# ── End-to-end training pipeline ─────────────────────────────────────────


def main() -> None:
    """Train Bidirectional LSTM at multiple sequence lengths on Mal-API-2019."""

    # ── Load cached data ─────────────────────────────────────────────────
    logger.info("Loading cached preprocessed data...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)
    vocab = load_json(cfg.VOCABULARY_PATH)
    label_encoder = load_pickle(cfg.CACHE_DIR / "label_encoder.pkl")

    vocab_size = len(vocab)
    num_classes = len(label_encoder.classes_)

    train_encoded = [s["encoded"] for s in train_samples]
    test_encoded = [s["encoded"] for s in test_samples]

    y_train_int = label_encoder.transform([s["label"] for s in train_samples])
    y_test_int = label_encoder.transform([s["label"] for s in test_samples])

    y_train_onehot = to_categorical(y_train_int, num_classes=num_classes)

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

        X_train = pad_sequences(train_encoded, max_len=seq_len)
        X_test = pad_sequences(test_encoded, max_len=seq_len)
        logger.info("Padded shapes: train=%s, test=%s.", X_train.shape, X_test.shape)

        model = build_lstm_model(vocab_size, seq_len, num_classes)
        model.summary(print_fn=lambda x: logger.info(x))

        model_path = cfg.LSTM_MODEL_DIR / f"best_len{seq_len}.keras"
        start = time.time()
        history = train_lstm(
            model, X_train, y_train_onehot, y_train_int, model_path,
        )
        elapsed = (time.time() - start) / 60.0
        logger.info("Training time for seq_len=%d: %.1f minutes.", seq_len, elapsed)

        # Evaluate on train set
        train_preds, _ = predict_with_confidence(model, X_train)
        train_acc = np.mean(train_preds == y_train_int)
        train_macro_f1 = f1_score(y_train_int, train_preds, average="macro")
        logger.info("seq_len=%d — Train accuracy: %.4f, Train macro-F1: %.4f",
                     seq_len, train_acc, train_macro_f1)

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

        best_val_loss = min(history.history.get("val_loss", [float("inf")]))
        best_val_acc = max(history.history.get("val_accuracy", [0.0]))
        epochs_trained = len(history.history["loss"])

        results[seq_len] = {
            "seq_len": seq_len,
            "train_accuracy": round(train_acc, 4),
            "train_macro_f1": round(train_macro_f1, 4),
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

    summary = {
        "best_seq_len": best_len,
        "all_results": {str(k): v for k, v in results.items()},
    }
    save_json(summary, cfg.METRICS_DIR / "lstm_comparison.json")
    logger.info("All LSTM results saved to %s.", cfg.METRICS_DIR / "lstm_comparison.json")


if __name__ == "__main__":
    main()
