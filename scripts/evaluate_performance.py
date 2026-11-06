# scripts/evaluate_performance.py
#
# Avaliação de desempenho para detecção comportamental UEBA.
# Compatível retroativamente com JSON anterior e estendido na Fase 6 com:
# - Average Precision (NaN se sem positivos)
# - Precision@K / Recall@K para K in {50, 100, 200}
# - MCC, Balanced Accuracy, F2-Score
# - Taxa operacional de alertas/dia e alertas/usuário

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from numpy.lib.stride_tricks import sliding_window_view


def _compute_precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int):
    n = len(scores)
    if n == 0:
        return 0.0, 0.0
    k_eff = min(k, n)
    top_indices = np.argsort(scores)[::-1][:k_eff]
    tp_k = int(np.sum(y_true[top_indices] == 1))
    prec_k = float(tp_k / k_eff) if k_eff > 0 else 0.0
    n_pos = int(np.sum(y_true == 1))
    rec_k = float(tp_k / n_pos) if n_pos > 0 else 0.0
    return prec_k, rec_k


def run_external_evaluation(config, test_df=None, scores=None, indices=None, verbose=False, y_true=None, users=None, y_pred=None):
    output_dir = config.get("output_dir", "./output")
    target_col = config.get("data", {}).get("target_col", "Is_Anomaly")
    sequence_length = config.get("model", {}).get("sequence_length", 10)
    thresh_mode = config.get("training", {}).get("threshold_mode", "per_user")

    # 1. Carrega scores e índices se não fornecidos em memória
    if scores is None or indices is None:
        scores_file = os.path.join(output_dir, "scores.npy") if os.path.exists(os.path.join(output_dir, "scores.npy")) else os.path.join(output_dir, "test_scores.npy")
        indices_file = os.path.join(output_dir, "indices.npy") if os.path.exists(os.path.join(output_dir, "indices.npy")) else os.path.join(output_dir, "test_indices.npy")
        scores = np.load(scores_file)
        indices = np.load(indices_file)

    # 2. Alinhamento Estrito de Rótulos (sempre com sort=False)
    if y_true is not None:
        y_test = np.asarray(y_true).astype(int)
    elif os.path.exists(os.path.join(output_dir, "y_true.npy")):
        y_test = np.load(os.path.join(output_dir, "y_true.npy")).astype(int)
    elif test_df is not None and not test_df.empty:
        y_test_list = []
        for user_id, group in test_df.groupby("UserID", sort=False):
            if len(group) < sequence_length:
                continue
            windows_labels = sliding_window_view(group[target_col].values, window_shape=sequence_length)
            y_test_list.append(np.max(windows_labels, axis=1))

        y_test = np.concatenate(y_test_list) if y_test_list else np.array([])
        y_test = np.nan_to_num(y_test, nan=0).astype(int)
    else:
        y_test = np.array([])

    if len(y_test) != len(scores):
        min_len = min(len(y_test), len(scores))
        y_test, scores = y_test[:min_len], scores[:min_len]

    # 3. Predições e Limiares
    if y_pred is not None:
        y_pred = np.asarray(y_pred).astype(int)
        if len(y_pred) != len(scores):
            y_pred = y_pred[:len(scores)]
    else:
        user_thresholds = {}
        thresholds_path = os.path.join(output_dir, "user_thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r", encoding="utf-8") as f:
                user_thresholds = json.load(f)

        global_fallback = user_thresholds.get("__GLOBAL_FALLBACK__", 0.5)
        if users is not None:
            test_users = [str(u) for u in users]
        else:
            user_mapping = test_df["UserID"].to_dict() if test_df is not None and "UserID" in test_df.columns else {}
            test_users = [str(user_mapping.get(idx, "unknown")) for idx in indices]

        sample_thresholds = np.array([
            user_thresholds.get(u, global_fallback) if thresh_mode == "per_user" and u != "__GLOBAL_FALLBACK__" else global_fallback 
            for u in test_users
        ])
        y_pred = (scores > sample_thresholds).astype(int)

    # 4. Cálculo das Métricas Canônicas e Novas Métricas (Fase 6)
    n_pos = int(np.sum(y_test == 1))
    
    # Average Precision (NaN se sem positivos)
    if n_pos > 0:
        avg_precision = float(average_precision_score(y_test, scores))
    else:
        avg_precision = None

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores) if len(np.unique(y_test)) > 1 else ([], [], [])
    pr_auc = float(auc(recall_curve, precision_curve)) if len(precision_curve) > 0 else 0.0
    try:
        roc_auc = float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) > 1 else 0.5
    except ValueError:
        roc_auc = 0.5 

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel() if len(y_test) > 0 else (0, 0, 0, 0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    class_1 = report_dict.get("1", report_dict.get("1.0", report_dict.get("True", {})))

    prec = float(class_1.get("precision", 0.0))
    rec = float(class_1.get("recall", 0.0))
    f1 = float(class_1.get("f1-score", 0.0))

    # F2-Score
    f2 = float((1.0 + 4.0) * (prec * rec) / ((4.0 * prec) + rec)) if (4.0 * prec + rec) > 0 else 0.0

    # MCC e Balanced Accuracy
    if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
        mcc = float(matthews_corrcoef(y_test, y_pred))
    else:
        mcc = 0.0
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred)) if len(np.unique(y_test)) > 1 else 0.5

    # Precision@K e Recall@K para K in {50, 100, 200}
    p50, r50 = _compute_precision_recall_at_k(y_test, scores, 50)
    p100, r100 = _compute_precision_recall_at_k(y_test, scores, 100)
    p200, r200 = _compute_precision_recall_at_k(y_test, scores, 200)

    # Taxas operacionais de alertas
    total_alerts = int(np.sum(y_pred))
    n_users = len(np.unique(users)) if users is not None and len(users) else (test_df["UserID"].nunique() if test_df is not None and "UserID" in test_df.columns else 1)
    alerts_per_user = float(total_alerts / max(n_users, 1))

    time_span_days = 1.0
    if test_df is not None and "Time" in test_df.columns and len(test_df) > 1:
        span_sec = float(test_df["Time"].max() - test_df["Time"].min())
        time_span_days = max(span_sec / 86400.0, 1.0 / 24.0)
    alerts_per_day = float(total_alerts / time_span_days)

    prevalence = float(np.mean(y_test)) if len(y_test) else 0.0
    lift_pr = float(pr_auc / prevalence) if prevalence > 0 else 0.0

    # Dicionário compatível com versões legadas
    metrics_dict = {
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "lift_pr": lift_pr,
        "roc_auc": roc_auc,
        "f1_score": f1,
        "f2_score": f2,
        "precision": prec,
        "recall": rec,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        # Novas métricas Fase 6
        "average_precision": avg_precision,
        "mcc": mcc,
        "balanced_accuracy": balanced_acc,
        "precision_at_50": p50,
        "recall_at_50": r50,
        "precision_at_100": p100,
        "recall_at_100": r100,
        "precision_at_200": p200,
        "recall_at_200": r200,
        "alerts_per_day": alerts_per_day,
        "alerts_per_user": alerts_per_user,
        "total_alerts": total_alerts,
    }

    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    if verbose:
        total_events = len(test_df) if test_df is not None else len(scores)
        anom_events = int(test_df[target_col].sum()) if test_df is not None and target_col in test_df.columns else n_pos
        prev_events = (anom_events / total_events * 100) if total_events > 0 else 0.0

        total_windows = len(y_test)
        anom_windows = int(np.sum(y_test))
        prev_windows = (anom_windows / total_windows * 100) if total_windows > 0 else 0.0

        print("\n" + "="*60)
        print("📊 RESUMO DA AUDITORIA DE DETECÇÃO (DOUBLE CHECK)")
        print("="*60)
        print(f" 🔹 Eventos Totais (Brutos)      : {total_events}")
        print(f" 🔹 Eventos Anômalos (Brutos)    : {anom_events} (Prevalência: {prev_events:.2f}%)")
        print("-" * 60)
        print(f" 🔹 Janelas Avaliadas (Seq={sequence_length})   : {total_windows}")
        print(f" 🔹 Janelas Anômalas (Gabarito)  : {anom_windows} (Prevalência: {prev_windows:.2f}%)")
        print(f" 🔹 Alertas Disparados           : {total_alerts} ({alerts_per_day:.1f}/dia, {alerts_per_user:.1f}/user)")
        print("-" * 60)
        print(f" ✅ Verdadeiros Positivos (TP)   : {int(tp)}")
        print(f" ⚠️  Falsos Positivos (FP)       : {int(fp)}")
        print(f" ❌ Falsos Negativos (FN)       : {int(fn)}")
        print(f" 🛡️  Verdadeiros Negativos (TN)   : {int(tn)}")
        print("-" * 60)
        print(f" 🎯 Precision                   : {prec:.4f} | Recall: {rec:.4f}")
        print(f" 🎯 F1-Score: {f1:.4f} | F2-Score: {f2:.4f} | MCC: {mcc:.4f}")
        print(f" 🎯 PR-AUC: {pr_auc:.4f} | Lift: {lift_pr:.2f}x | Avg Precision: {avg_precision}")
        print(f" 🎯 Precision@50: {p50:.4f} | Recall@50: {r50:.4f}")
        print(f" 🎯 Precision@100: {p100:.4f} | Recall@100: {r100:.4f}")
        print(f" 🎯 ROC-AUC                     : {roc_auc:.4f}")
        print("="*60 + "\n")
        
    return metrics_dict
