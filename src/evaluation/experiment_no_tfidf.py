"""
Experiment: evaluate XGBoost generalizability using ONLY the 81 engineered
features (9 statistical + 8 category + 64 bigram), dropping all 5000 TF-IDF
features.

Hypothesis: TF-IDF features are the primary source of cross-sandbox overfit
because they encode Cuckoo-specific API token frequencies that don't transfer
to CAPE traces (20% UNK ratio, completely different call distributions).

This script:
  1. Loads V2 training data and builds 81-feature matrices (no TF-IDF).
  2. Trains a fresh XGBoost with the V2 regularised param dist.
  3. Evaluates on Mal-API test set (baseline), MalBehavD, and WinMET.
  4. Compares macro-F1 against the full 5081-feature V2 model.

Usage::

    python -m src.evaluation.experiment_no_tfidf
"""

import time

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import config as cfg
from src.data_loading.preprocessing import (
    load_winmet_samples,
    preprocess_external_samples,
)
from src.evaluation.metrics import compute_all_metrics, plot_confusion_matrix
from src.model_training.feature_engineering import (
    compute_bigram_transition_features,
    compute_category_features,
    compute_statistical_features,
)
from src.model_training.xgboost_model import (
    predict_with_confidence,
    train_xgboost,
)
from src.utils import get_logger, load_json, load_pickle, save_json

logger = get_logger(__name__)


def build_engineered_features(
    samples: list,
    log_dampen: bool = True,
) -> np.ndarray:
    """Build the 81-feature matrix (no TF-IDF).

    Args:
        samples: Samples with ``sequence`` field.
        log_dampen: Whether to log-dampen category/bigram/stat features.

    Returns:
        Dense array of shape ``(n_samples, 81)``.
    """
    stats = compute_statistical_features(samples, log_dampen=log_dampen)
    cats = compute_category_features(samples, log_dampen=log_dampen)
    bigrams = compute_bigram_transition_features(samples, log_dampen=log_dampen)
    combined = np.hstack([stats, cats, bigrams])
    logger.info(
        "Engineered features only: shape %s (stats=%d, cats=%d, bigrams=%d).",
        combined.shape, stats.shape[1], cats.shape[1], bigrams.shape[1],
    )
    return combined


def main() -> None:
    out_dir = cfg.RESULTS_DIR / "v2" / "WinMET" / "experiment_no_tfidf"
    metrics_dir = out_dir / "metrics"
    plots_dir = out_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load V2 artifacts ───────────────────────────────────────────────
    label_encoder: LabelEncoder = load_pickle(cfg.V2_LABEL_ENCODER_PATH)
    class_names = list(label_encoder.classes_)
    vocab = load_json(cfg.VOCABULARY_PATH)

    # ── Training data ───────────────────────────────────────────────────
    logger.info("Loading V2 training data (8-class, with Trojan)...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)

    y_train = label_encoder.transform([s["label"] for s in train_samples])
    y_test = label_encoder.transform([s["label"] for s in test_samples])

    X_train = build_engineered_features(train_samples)
    X_test = build_engineered_features(test_samples)

    logger.info("Feature matrix: train=%s, test=%s.", X_train.shape, X_test.shape)

    # ── Train XGBoost (81 features) ─────────────────────────────────────
    logger.info("Training XGBoost on 81 engineered features...")
    start = time.time()
    model, best_params = train_xgboost(
        X_train, y_train, label_encoder,
        param_dist=cfg.XGBOOST_V2_PARAM_DIST,
    )
    elapsed_min = (time.time() - start) / 60.0
    logger.info("Training completed in %.1f minutes.", elapsed_min)

    # ── Evaluate: Mal-API test (in-distribution) ────────────────────────
    logger.info("=" * 60)
    logger.info("MAL-API TEST SET (81 features)")
    logger.info("=" * 60)

    preds, probs = predict_with_confidence(model, X_test)
    malapi_metrics = compute_all_metrics(y_test, preds, probs, class_names)
    save_json(malapi_metrics, metrics_dir / "malapi_test_xgboost_81feat.json")
    logger.info(
        "Mal-API: acc=%.4f macro-F1=%.4f",
        malapi_metrics["accuracy"], malapi_metrics["macro_f1"],
    )
    logger.info(
        "\n%s",
        classification_report(
            y_test, preds,
            target_names=class_names, zero_division=0,
        ),
    )
    plot_confusion_matrix(
        y_test, preds, class_names,
        "XGBoost 81-feat — Mal-API Test",
        plots_dir / "malapi_test_xgboost_81feat_confusion.png",
    )

    # ── Evaluate: MalBehavD ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MALBEHAVD (81 features)")
    logger.info("=" * 60)

    mb_data = load_json(cfg.NO_TROJAN_MALBEHAVD_PATH)
    mb_malware = [s for s in mb_data["samples"] if s["label"] != cfg.BENIGN_LABEL]
    known = set(class_names)
    mb_malware = [s for s in mb_malware if s["label"] in known]

    mb_v2 = preprocess_external_samples(
        mb_malware, vocab, normalize_for_vocab=False,
        dataset_name="MalBehavD (81-feat experiment)",
    )

    y_mb = label_encoder.transform([s["label"] for s in mb_v2])
    X_mb = build_engineered_features(mb_v2)
    mb_preds, mb_probs = predict_with_confidence(model, X_mb)
    mb_metrics = compute_all_metrics(y_mb, mb_preds, mb_probs, class_names)
    save_json(mb_metrics, metrics_dir / "malbehavd_xgboost_81feat.json")
    logger.info(
        "MalBehavD: acc=%.4f macro-F1=%.4f",
        mb_metrics["accuracy"], mb_metrics["macro_f1"],
    )
    plot_confusion_matrix(
        y_mb, mb_preds, class_names,
        "XGBoost 81-feat — MalBehavD",
        plots_dir / "malbehavd_xgboost_81feat_confusion.png",
    )

    # ── Evaluate: WinMET ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("WINMET (81 features, cross-sandbox)")
    logger.info("=" * 60)

    wm_raw = load_winmet_samples(drop_trojan=True)
    wm_raw = [s for s in wm_raw if s["label"] in known]
    wm_v2 = preprocess_external_samples(
        wm_raw, vocab, normalize_for_vocab=True,
        dataset_name="WinMET (81-feat experiment)",
    )

    y_wm = label_encoder.transform([s["label"] for s in wm_v2])
    X_wm = build_engineered_features(wm_v2)
    wm_preds, wm_probs = predict_with_confidence(model, X_wm)
    wm_metrics = compute_all_metrics(y_wm, wm_preds, wm_probs, class_names)
    save_json(wm_metrics, metrics_dir / "winmet_xgboost_81feat.json")
    logger.info(
        "WinMET: acc=%.4f macro-F1=%.4f",
        wm_metrics["accuracy"], wm_metrics["macro_f1"],
    )
    logger.info(
        "\n%s",
        classification_report(
            y_wm, wm_preds,
            target_names=class_names, zero_division=0,
        ),
    )
    plot_confusion_matrix(
        y_wm, wm_preds, class_names,
        "XGBoost 81-feat — WinMET",
        plots_dir / "winmet_xgboost_81feat_confusion.png",
    )

    # ── Comparison table ────────────────────────────────────────────────
    # Load V2 full-feature results for comparison
    v2_malapi_path = cfg.V2_METRICS_DIR / "malapi_test_xgboost.json"
    v2_winmet_path = cfg.V2_WINMET_METRICS_DIR / "winmet_xgboost.json"
    v2_mb_path = cfg.V2_MALBEHAVD_METRICS_DIR / "malbehavd_xgboost.json"

    v2_malapi = load_json(v2_malapi_path) if v2_malapi_path.exists() else None
    v2_winmet = load_json(v2_winmet_path) if v2_winmet_path.exists() else None
    v2_mb = load_json(v2_mb_path) if v2_mb_path.exists() else None

    comparison = {
        "experiment": "81 engineered features only (no TF-IDF)",
        "training_minutes": round(elapsed_min, 1),
        "best_params": best_params,
        "results": {
            "malapi": {
                "81feat_macro_f1": malapi_metrics["macro_f1"],
                "81feat_accuracy": malapi_metrics["accuracy"],
                "full_macro_f1": v2_malapi["macro_f1"] if v2_malapi else None,
                "full_accuracy": v2_malapi["accuracy"] if v2_malapi else None,
            },
            "malbehavd": {
                "81feat_macro_f1": mb_metrics["macro_f1"],
                "81feat_accuracy": mb_metrics["accuracy"],
                "full_macro_f1": v2_mb["macro_f1"] if v2_mb else None,
                "full_accuracy": v2_mb["accuracy"] if v2_mb else None,
            },
            "winmet": {
                "81feat_macro_f1": wm_metrics["macro_f1"],
                "81feat_accuracy": wm_metrics["accuracy"],
                "full_macro_f1": v2_winmet["macro_f1"] if v2_winmet else None,
                "full_accuracy": v2_winmet["accuracy"] if v2_winmet else None,
            },
        },
    }
    save_json(comparison, metrics_dir / "comparison_81feat_vs_full.json")

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Experiment: 81 Engineered Features vs. Full 5081 Features (XGBoost)")
    print(f"{'='*65}")
    print(f"Training time: {elapsed_min:.1f} min | Best params: {best_params}")
    print()
    print(f"  {'Dataset':<12} {'81-feat F1':>11} {'Full F1':>9} {'Delta':>8} {'Direction':>10}")
    print(f"  {'-'*52}")

    for ds_name, feat81, full_src in [
        ("Mal-API", malapi_metrics, v2_malapi),
        ("MalBehavD", mb_metrics, v2_mb),
        ("WinMET", wm_metrics, v2_winmet),
    ]:
        f1_81 = feat81["macro_f1"]
        if full_src:
            f1_full = full_src["macro_f1"]
            delta = f1_81 - f1_full
            direction = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
            print(
                f"  {ds_name:<12} {f1_81:>11.4f} {f1_full:>9.4f} "
                f"{delta:>+8.4f} {direction:>10}"
            )
        else:
            print(f"  {ds_name:<12} {f1_81:>11.4f} {'N/A':>9}")

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
