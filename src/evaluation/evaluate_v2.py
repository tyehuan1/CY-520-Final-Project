"""
V2 model evaluation: Mal-API test set + generalizability on MalBehavD & WinMET.

Runs all three V2 models (XGBoost, LSTM, Ensemble) on:
  1. Mal-API-2019 test set (with Trojan, 8-class) — in-distribution baseline
  2. MalBehavD-V1 malware — cross-dataset generalizability (Cuckoo→Cuckoo)
  3. WinMET no-Trojan — cross-sandbox generalizability (Cuckoo→CAPE)

For LSTM evaluations on long sequences, uses sliding-window inference so
the full sequence is seen (not just the first 500 tokens).

Usage::

    python -m src.evaluation.evaluate_v2
"""

import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import config as cfg
from src.evaluation.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_f1,
    plot_roc_curves,
    plot_three_model_comparison,
    run_shap_analysis,
)
from src.model_training.feature_engineering import build_feature_matrix
from src.model_training.lstm_model import (
    load_model as load_lstm_model,
    predict_with_confidence as lstm_predict,
    predict_with_sliding_window,
)
from src.model_training.xgboost_model import (
    load_model as load_xgb_model,
    predict_with_confidence as xgb_predict,
)
from src.data_loading.preprocessing import (
    compute_unk_ratio,
    pad_sequences,
    preprocess_external_samples,
)
from src.utils import get_logger, load_json, load_pickle, save_json

logger = get_logger(__name__)


def _run_model_suite(
    tag: str,
    samples,
    y_true,
    class_names,
    label_encoder,
    xgb_model,
    lstm_model,
    ensemble,
    tfidf_vec,
    plots_dir,
    metrics_dir,
    use_sliding_window: bool = False,
):
    """Run XGBoost, LSTM, Ensemble on a dataset and save results.

    Returns dict of {model_name: metrics_dict}.
    Also stores raw predictions/probabilities in the returned dict under
    ``_preds`` and ``_probs`` keys (prefixed with underscore so they are
    clearly internal and not serialised).
    """
    results = {}

    # ── XGBoost ──
    X_xgb = build_feature_matrix(samples, tfidf_vec, log_dampen=True)
    xgb_preds, xgb_probs = xgb_predict(xgb_model, X_xgb)
    xgb_metrics = compute_all_metrics(y_true, xgb_preds, xgb_probs, class_names)
    results["xgboost"] = xgb_metrics
    results["_xgb_preds"] = xgb_preds
    results["_xgb_probs"] = xgb_probs
    results["_X_xgb"] = X_xgb
    save_json(xgb_metrics, metrics_dir / f"{tag}_xgboost.json")
    logger.info(
        "%s XGBoost: acc=%.4f macro-F1=%.4f",
        tag, xgb_metrics["accuracy"], xgb_metrics["macro_f1"],
    )
    logger.info(
        "%s XGBoost report:\n%s", tag,
        classification_report(
            y_true, xgb_preds,
            labels=list(range(len(class_names))),
            target_names=class_names, zero_division=0,
        ),
    )
    plot_confusion_matrix(
        y_true, xgb_preds, class_names,
        f"V2 XGBoost — {tag}",
        plots_dir / f"{tag}_xgboost_confusion.png",
    )
    plot_per_class_f1(
        xgb_metrics,
        f"V2 XGBoost Per-Class F1 — {tag}",
        plots_dir / f"{tag}_xgboost_per_class_f1.png",
    )
    try:
        plot_roc_curves(
            y_true, xgb_probs, class_names,
            f"V2 XGBoost ROC — {tag}",
            plots_dir / f"{tag}_xgboost_roc.png",
        )
    except ValueError as e:
        logger.warning("Could not plot XGBoost ROC for %s: %s", tag, e)

    # ── LSTM ──
    if use_sliding_window:
        encoded_seqs = [s["encoded"] for s in samples]
        lstm_preds, lstm_probs = predict_with_sliding_window(
            lstm_model, encoded_seqs,
            window_len=cfg.LSTM_V2_BEST_SEQ_LEN,
            stride=cfg.LSTM_V2_SLIDING_STRIDE,
        )
    else:
        X_lstm = pad_sequences(
            [s["encoded"] for s in samples], max_len=cfg.LSTM_V2_BEST_SEQ_LEN,
        )
        lstm_preds, lstm_probs = lstm_predict(lstm_model, X_lstm)

    lstm_metrics = compute_all_metrics(y_true, lstm_preds, lstm_probs, class_names)
    results["lstm"] = lstm_metrics
    results["_lstm_preds"] = lstm_preds
    results["_lstm_probs"] = lstm_probs
    save_json(lstm_metrics, metrics_dir / f"{tag}_lstm.json")
    logger.info(
        "%s LSTM: acc=%.4f macro-F1=%.4f",
        tag, lstm_metrics["accuracy"], lstm_metrics["macro_f1"],
    )
    plot_confusion_matrix(
        y_true, lstm_preds, class_names,
        f"V2 LSTM — {tag}",
        plots_dir / f"{tag}_lstm_confusion.png",
    )
    plot_per_class_f1(
        lstm_metrics,
        f"V2 LSTM Per-Class F1 — {tag}",
        plots_dir / f"{tag}_lstm_per_class_f1.png",
    )
    try:
        plot_roc_curves(
            y_true, lstm_probs, class_names,
            f"V2 LSTM ROC — {tag}",
            plots_dir / f"{tag}_lstm_roc.png",
        )
    except ValueError as e:
        logger.warning("Could not plot LSTM ROC for %s: %s", tag, e)

    # ── Ensemble ──
    if ensemble is not None:
        ens_preds, ens_probs = ensemble.predict_from_precomputed(
            xgb_probs, lstm_probs,
        )
        ens_metrics = compute_all_metrics(y_true, ens_preds, ens_probs, class_names)
        results["ensemble"] = ens_metrics
        results["_ens_preds"] = ens_preds
        results["_ens_probs"] = ens_probs
        save_json(ens_metrics, metrics_dir / f"{tag}_ensemble.json")
        logger.info(
            "%s Ensemble: acc=%.4f macro-F1=%.4f",
            tag, ens_metrics["accuracy"], ens_metrics["macro_f1"],
        )
        plot_confusion_matrix(
            y_true, ens_preds, class_names,
            f"V2 Ensemble — {tag}",
            plots_dir / f"{tag}_ensemble_confusion.png",
        )
        plot_per_class_f1(
            ens_metrics,
            f"V2 Ensemble Per-Class F1 — {tag}",
            plots_dir / f"{tag}_ensemble_per_class_f1.png",
        )
        try:
            plot_roc_curves(
                y_true, ens_probs, class_names,
                f"V2 Ensemble ROC — {tag}",
                plots_dir / f"{tag}_ensemble_roc.png",
            )
        except ValueError as e:
            logger.warning("Could not plot Ensemble ROC for %s: %s", tag, e)

    # ── Model comparison charts ──
    if ensemble is not None:
        plot_three_model_comparison(
            xgb_metrics, lstm_metrics, ens_metrics,
            plots_dir / f"{tag}_model_comparison.png",
        )
    else:
        plot_model_comparison(
            xgb_metrics, lstm_metrics,
            plots_dir / f"{tag}_model_comparison.png",
        )

    return results


def _secondary_class_hit_rate(samples, y_true, y_pred, class_names):
    """For wrong predictions, what fraction land on a *secondary* class?"""
    name_by_idx = {i: n for i, n in enumerate(class_names)}
    n_total = len(samples)
    n_wrong = 0
    n_with_secondary = 0
    n_secondary_hit = 0

    per_family = {}
    for s, yt, yp in zip(samples, y_true, y_pred):
        fam = s.get("family_avclass", "<unknown>")
        secondary = s.get("secondary_classes", []) or []
        pf = per_family.setdefault(
            fam,
            {"n": 0, "wrong": 0, "secondary_defined": 0,
             "secondary_hit": 0, "primary_hit": 0},
        )
        pf["n"] += 1
        if secondary:
            pf["secondary_defined"] += 1
        if yt == yp:
            pf["primary_hit"] += 1
            continue
        n_wrong += 1
        pf["wrong"] += 1
        if not secondary:
            continue
        n_with_secondary += 1
        if name_by_idx.get(int(yp)) in secondary:
            n_secondary_hit += 1
            pf["secondary_hit"] += 1

    return {
        "n_total": n_total,
        "n_correct_primary": n_total - n_wrong,
        "n_wrong": n_wrong,
        "n_wrong_with_secondary_defined": n_with_secondary,
        "n_secondary_hit": n_secondary_hit,
        "secondary_hit_rate_over_wrong": (
            round(n_secondary_hit / n_wrong, 4) if n_wrong else None
        ),
        "secondary_hit_rate_over_eligible": (
            round(n_secondary_hit / n_with_secondary, 4)
            if n_with_secondary else None
        ),
        "primary_or_secondary_hit_rate": round(
            (n_total - n_wrong + n_secondary_hit) / n_total, 4,
        ) if n_total else None,
        "per_family": per_family,
    }


def _misclassification_analysis(y_true, y_pred, class_names):
    """Build a dict of where each class's errors go."""
    analysis = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        preds_for_class = y_pred[mask]
        total = int(mask.sum())
        correct = int((preds_for_class == i).sum())
        wrong = total - correct
        if wrong == 0:
            analysis[name] = {"total": total, "correct": correct, "errors": {}}
            continue
        wrong_preds = preds_for_class[preds_for_class != i]
        error_dist = Counter(wrong_preds)
        errors = {
            class_names[k]: int(v)
            for k, v in error_dist.most_common()
        }
        analysis[name] = {"total": total, "correct": correct, "errors": errors}
    return analysis


def main() -> None:
    malapi_plots_dir = cfg.V2_PLOTS_DIR
    malapi_metrics_dir = cfg.V2_METRICS_DIR
    mb_plots_dir = cfg.V2_MALBEHAVD_PLOTS_DIR
    mb_metrics_dir = cfg.V2_MALBEHAVD_METRICS_DIR
    wm_plots_dir = cfg.V2_WINMET_PLOTS_DIR
    wm_metrics_dir = cfg.V2_WINMET_METRICS_DIR
    shap_dir = cfg.V2_RESULTS_DIR / "MalAPI" / "shap"
    for d in [malapi_plots_dir, malapi_metrics_dir, mb_plots_dir,
              mb_metrics_dir, wm_plots_dir, wm_metrics_dir, shap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Load V2 artifacts
    # ==================================================================
    logger.info("Loading V2 model artifacts...")
    xgb_model = load_xgb_model(cfg.V2_XGBOOST_MODEL_DIR / "best_model.pkl")
    lstm_model = load_lstm_model(
        cfg.V2_LSTM_MODEL_DIR / f"best_len{cfg.LSTM_V2_BEST_SEQ_LEN}.keras",
    )
    tfidf_vec = load_pickle(cfg.V2_TFIDF_PATH)
    label_encoder = load_pickle(cfg.V2_LABEL_ENCODER_PATH)
    class_names = list(label_encoder.classes_)
    vocab = load_json(cfg.VOCABULARY_PATH)

    # Load ensemble
    from src.model_training.ensemble_model import EnsembleClassifier, load_model as load_ensemble
    import __main__
    __main__.EnsembleClassifier = EnsembleClassifier
    ens_path = cfg.V2_ENSEMBLE_MODEL_DIR / "ensemble_model.pkl"
    ensemble = load_ensemble(ens_path) if ens_path.exists() else None

    # ==================================================================
    # 1. Mal-API test set (in-distribution, 8-class)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("V2 EVALUATION — MAL-API TEST SET (8-class)")
    logger.info("=" * 70)

    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)
    y_test = label_encoder.transform([s["label"] for s in test_samples])

    malapi_results = _run_model_suite(
        "malapi_test", test_samples, y_test, class_names,
        label_encoder, xgb_model, lstm_model, ensemble, tfidf_vec,
        malapi_plots_dir, malapi_metrics_dir,
        use_sliding_window=False,  # Mal-API seqs are short enough
    )

    # Save comparison
    malapi_comparison = {
        name: {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        for name, m in malapi_results.items()
        if not name.startswith("_")
    }
    save_json(malapi_comparison, malapi_metrics_dir / "malapi_model_comparison.json")

    # ── SHAP analysis (XGBoost, Mal-API test set) ────────────────────
    logger.info("=" * 70)
    logger.info("V2 SHAP ANALYSIS")
    logger.info("=" * 70)

    feature_names = tfidf_vec.get_feature_names_out().tolist()
    from src.model_training.feature_engineering import (
        STATISTICAL_FEATURE_NAMES,
        CATEGORY_FEATURE_NAMES,
        BIGRAM_FEATURE_NAMES,
    )
    feature_names = feature_names + list(STATISTICAL_FEATURE_NAMES) + CATEGORY_FEATURE_NAMES + BIGRAM_FEATURE_NAMES
    X_xgb_test = malapi_results["_X_xgb"]
    run_shap_analysis(
        xgb_model, X_xgb_test, feature_names, class_names, shap_dir,
    )

    # ==================================================================
    # 2. MalBehavD-V1 (malware only, no Trojan)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("V2 GENERALIZABILITY — MALBEHAVD (malware-only)")
    logger.info("=" * 70)

    mb_data = load_json(cfg.NO_TROJAN_MALBEHAVD_PATH)
    all_mb = mb_data["samples"]
    mb_malware = [s for s in all_mb if s["label"] != cfg.BENIGN_LABEL]

    # Filter to classes this model knows (MalBehavD has no Trojan after
    # no-Trojan filtering, but guard against unknowns)
    known = set(class_names)
    mb_malware = [s for s in mb_malware if s["label"] in known]
    logger.info("MalBehavD malware: %d samples.", len(mb_malware))

    # MalBehavD samples from build_no_trojan are encoded with the
    # no-Trojan vocab.  V2 uses the with-Trojan vocab.  Re-encode.
    mb_malware_v2 = preprocess_external_samples(
        mb_malware, vocab, normalize_for_vocab=False,
        dataset_name="MalBehavD-V1 (v2 vocab)",
    )

    y_mb = label_encoder.transform([s["label"] for s in mb_malware_v2])
    mb_results = _run_model_suite(
        "malbehavd", mb_malware_v2, y_mb, class_names,
        label_encoder, xgb_model, lstm_model, ensemble, tfidf_vec,
        mb_plots_dir, mb_metrics_dir,
        use_sliding_window=False,  # MalBehavD seqs are very short
    )

    # ==================================================================
    # 3. WinMET (no-Trojan, cross-sandbox)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("V2 GENERALIZABILITY — WINMET (no-Trojan, CAPE)")
    logger.info("=" * 70)

    # Load pre-encoded WinMET from build_no_trojan (encoded with no-Trojan
    # vocab).  Re-encode with the with-Trojan vocab + normalization.
    from src.data_loading.preprocessing import load_winmet_samples
    wm_raw = load_winmet_samples(drop_trojan=True)
    wm_raw = [s for s in wm_raw if s["label"] in known]
    logger.info("WinMET (no-Trojan): %d samples.", len(wm_raw))

    wm_v2 = preprocess_external_samples(
        wm_raw, vocab, normalize_for_vocab=True,
        dataset_name="WinMET (v2 vocab)",
    )

    wm_unk = compute_unk_ratio(wm_v2, vocab)
    logger.info("WinMET UNK ratio (v2 vocab): %.2f%%", wm_unk * 100)

    y_wm = label_encoder.transform([s["label"] for s in wm_v2])

    # Use sliding-window for LSTM — WinMET median seq len is 5379
    wm_results = _run_model_suite(
        "winmet", wm_v2, y_wm, class_names,
        label_encoder, xgb_model, lstm_model, ensemble, tfidf_vec,
        wm_plots_dir, wm_metrics_dir,
        use_sliding_window=True,
    )

    # ── Secondary-class hit rate (WinMET) ────────────────────────────
    logger.info("-" * 70)
    logger.info("Secondary-class hit-rate analysis (WinMET)")
    logger.info("-" * 70)

    # Use cached predictions from _run_model_suite
    wm_xgb_preds = wm_results["_xgb_preds"]
    wm_xgb_probs = wm_results["_xgb_probs"]
    wm_lstm_preds = wm_results["_lstm_preds"]
    wm_lstm_probs = wm_results["_lstm_probs"]

    secondary_analysis = {}
    preds_map = [("xgboost", wm_xgb_preds)]
    preds_map.append(("lstm", wm_lstm_preds))
    if ensemble is not None:
        wm_ens_preds = wm_results["_ens_preds"]
        preds_map.append(("ensemble", wm_ens_preds))

    for model_name, preds in preds_map:
        sa = _secondary_class_hit_rate(wm_v2, y_wm, preds, class_names)
        secondary_analysis[model_name] = sa
        logger.info(
            "  %s: wrong=%d, sec-hit=%d (%.2f%% of wrong). "
            "Primary-or-secondary acc=%.2f%%",
            model_name, sa["n_wrong"], sa["n_secondary_hit"],
            100 * (sa["secondary_hit_rate_over_wrong"] or 0),
            100 * (sa["primary_or_secondary_hit_rate"] or 0),
        )
    save_json(secondary_analysis, wm_metrics_dir / "winmet_secondary_class_analysis.json")

    # ── Misclassification analysis ───────────────────────────────────
    logger.info("-" * 70)
    logger.info("Misclassification analysis")
    logger.info("-" * 70)

    misclass = {}
    for dataset_tag, preds, y, samps in [
        ("winmet_xgboost", wm_xgb_preds, y_wm, wm_v2),
        ("winmet_lstm", wm_lstm_preds, y_wm, wm_v2),
    ]:
        ma = _misclassification_analysis(y, preds, class_names)
        misclass[dataset_tag] = ma
        logger.info("  %s:", dataset_tag)
        for cls_name, info in ma.items():
            if info["errors"]:
                top_errors = list(info["errors"].items())[:3]
                err_str = ", ".join(f"{k}={v}" for k, v in top_errors)
                logger.info(
                    "    %12s: %d/%d correct — top errors: %s",
                    cls_name, info["correct"], info["total"], err_str,
                )
    save_json(misclass, wm_metrics_dir / "misclassification_analysis.json")

    # ==================================================================
    # Generalization Gap
    # ==================================================================
    logger.info("=" * 70)
    logger.info("V2 GENERALIZATION GAP")
    logger.info("=" * 70)

    gap = {}
    for model_name in ["xgboost", "lstm"] + (["ensemble"] if ensemble else []):
        ma_f1 = malapi_results[model_name]["macro_f1"]
        mb_f1 = mb_results[model_name]["macro_f1"]
        wm_f1 = wm_results[model_name]["macro_f1"]
        gap[model_name] = {
            "malapi_f1": ma_f1,
            "malbehavd_f1": mb_f1,
            "winmet_f1": wm_f1,
            "malbehavd_gap": round(ma_f1 - mb_f1, 4),
            "winmet_gap": round(ma_f1 - wm_f1, 4),
            "malbehavd_drop_pct": round(100 * (ma_f1 - mb_f1) / ma_f1, 1) if ma_f1 > 0 else None,
            "winmet_drop_pct": round(100 * (ma_f1 - wm_f1) / ma_f1, 1) if ma_f1 > 0 else None,
        }
        logger.info(
            "%s: MalAPI=%.4f | MalBehavD=%.4f (%.1f%% drop) | WinMET=%.4f (%.1f%% drop)",
            model_name.upper(), ma_f1,
            mb_f1, gap[model_name]["malbehavd_drop_pct"] or 0,
            wm_f1, gap[model_name]["winmet_drop_pct"] or 0,
        )
    save_json(gap, wm_metrics_dir / "generalization_gap.json")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("V2 Evaluation Summary")
    print(f"{'='*70}")

    print(f"\nUNK ratio (WinMET, v2 vocab): {wm_unk*100:.2f}%")

    print(f"\n--- Macro-F1 by dataset ---")
    print(f"  {'Model':<10} {'MalAPI':>9} {'MalBehD':>9} {'Drop%':>7} {'WinMET':>9} {'Drop%':>7}")
    print(f"  {'-'*55}")
    for name in ["xgboost", "lstm"] + (["ensemble"] if ensemble else []):
        g = gap[name]
        print(
            f"  {name:<10} {g['malapi_f1']:>9.4f} {g['malbehavd_f1']:>9.4f} "
            f"{g['malbehavd_drop_pct']:>6.1f}% {g['winmet_f1']:>9.4f} "
            f"{g['winmet_drop_pct']:>6.1f}%"
        )

    print(f"\n--- Secondary-Class Hit Rate (WinMET) ---")
    print(f"  {'Model':<10} {'Wrong':>7} {'SecHit':>7} {'%Wrong':>8} {'Pri+Sec':>8}")
    print(f"  {'-'*45}")
    for name, sa in secondary_analysis.items():
        pw = 100 * (sa["secondary_hit_rate_over_wrong"] or 0)
        pa = 100 * (sa["primary_or_secondary_hit_rate"] or 0)
        print(f"  {name:<10} {sa['n_wrong']:>7d} {sa['n_secondary_hit']:>7d} {pw:>7.2f}% {pa:>7.2f}%")

    print(f"\nResults saved to: {cfg.V2_RESULTS_DIR}")


if __name__ == "__main__":
    main()
