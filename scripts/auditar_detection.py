#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auditar_detection.py — reproduz o bloco:
============================================================
📊 RESUMO DA AUDITORIA DE DETECÇÃO (DOUBLE CHECK)
============================================================
Lê os artefatos REAIS de um run_dir e reaplica Trainer._build_sequences +
_calculate_scores + user_thresholds.json.
"""

import os
import sys
import json
import pickle
import argparse
import glob
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from core.data.dataset_adapter import DatasetAdapter
from core.features.feature_processor import FeatureProcessor
from core.temporal_builder import TemporalBuilder
from core.trainer import Trainer
from core.models.model_factory import create as create_model


# =====================================================================
# CARGA DOS ARTEFATOS
# =====================================================================
def _load_run(run_dir):
    cfg = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(cfg):
        raise FileNotFoundError(f"config_used.yaml ausente em {run_dir}")
    with open(cfg, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    meta = {}
    mp = os.path.join(run_dir, "trainer_meta.pkl")
    if os.path.exists(mp):
        with open(mp, "rb") as f:
            meta = pickle.load(f)

    thresholds = {}
    tp = os.path.join(run_dir, "user_thresholds.json")
    if os.path.exists(tp):
        with open(tp) as f:
            thresholds = json.load(f)

    model = create_model(config)
    model_dir = os.path.join(run_dir, "saved_model")
    if hasattr(model, "load") and os.path.isdir(model_dir):
        try:
            model.load(model_dir)
        except Exception as e:
            print(f"⚠️  Falha ao carregar modelo ({e}).")
    return config, model, meta, thresholds


# =====================================================================
# DATASET
# =====================================================================
def _resolve_test_df(config, run_dir, data_override):
    if data_override:
        path = data_override
    else:
        cand = os.path.join(run_dir, "test_audit.csv")
        if os.path.exists(cand):
            path = cand
        else:
            path = (config.get("data", {}).get("dataset_path")
                    or config.get("data", {}).get("test_path"))
            if not path:
                raise FileNotFoundError("Nenhum dataset resolvível. Passe --data.")

    adapter = DatasetAdapter(config)
    raw = adapter._load_resource(path)
    if raw is None or raw.empty:
        raise ValueError(f"Dataset vazio: {path}")
    raw = adapter._normalize(raw)
    raw = adapter._apply_auto_labels(raw, path)

    if not path.endswith("test_audit.csv"):
        _, _, raw = adapter._perform_dynamic_split(raw)
    return raw, path


# =====================================================================
# PIPELINE + JANELAS
# =====================================================================
def _build_windows(t_df, config, meta, model):
    trainer = Trainer(config, model)
    trainer.scalers = meta.get("scalers", {}) or {}
    trainer.feature_columns = meta.get("feature_columns", None)

    X, indices, users, y_true = trainer._build_sequences(
        t_df, fit=False, return_labels=True
    )
    if X is None or len(X) == 0:
        raise ValueError("Nenhuma janela gerada.")
    scores = trainer._calculate_scores(X)
    seq_len = config.get("model", {}).get("sequence_length", 10)
    return trainer, X, indices, users, y_true, scores, seq_len


def _thresholds_per_user(users, scores, thresholds):
    fb = thresholds.get("__GLOBAL_FALLBACK__")
    if fb is None:
        fb = float(np.median(scores))
    thr = np.array([thresholds.get(str(u), fb) for u in users])
    return (scores > thr).astype(int)


# =====================================================================
# RELATÓRIO
# =====================================================================
def _print_report(total_events, total_anomalies, seq_len, y_true, y_pred, scores):
    tot_w = len(y_true)
    anom_w = int(np.sum(y_true))
    alerts = int(np.sum(y_pred))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    try:
        pr = average_precision_score(y_true, scores)
    except Exception:
        pr = 0.0
    try:
        roc = roc_auc_score(y_true, scores)
    except Exception:
        roc = 0.0
    raw_prev = total_anomalies / total_events if total_events else 0.0
    win_prev = anom_w / tot_w if tot_w else 0.0

    print("=" * 60)
    print("📊 RESUMO DA AUDITORIA DE DETECÇÃO (DOUBLE CHECK)")
    print("=" * 60)
    print(f" 🔹 Eventos Totais (Brutos)      : {total_events:,}")
    print(f" 🔹 Eventos Anômalos (Brutos)    : {total_anomalies:,} (Prevalência: {raw_prev*100:.2f}%)")
    print("-" * 60)
    print(f" 🔹 Janelas Avaliadas (Seq={seq_len})   : {tot_w:,}")
    print(f" 🔹 Janelas Anômalas (Gabarito)  : {anom_w:,} (Prevalência: {win_prev*100:.2f}%)")
    print(f" 🔹 Alertas Disparados           : {alerts:,}")
    print("-" * 60)
    print(f" ✅ Verdadeiros Positivos (TP) : {tp:,}")
    print(f" ⚠️  Falsos Positivos (FP)     : {fp:,}")
    print(f" ❌ Falsos Negativos (FN)     : {fn:,}")
    print(f" 🛡️  Verdadeiros Negativos (TN) : {tn:,}")
    print("-" * 60)
    print(f" 🎯 Precision (Precisão)      : {prec:.4f}")
    print(f" 🎯 Recall (Sensibilidade)    : {rec:.4f}")
    print(f" 🎯 F1-Score                  : {f1:.4f}")
    print(f" 🎯 PR-AUC                    : {pr:.4f}")
    print(f" 🎯 ROC-AUC                   : {roc:.4f}")
    print("=" * 60)
    return dict(events=total_events, anomalies_events=total_anomalies,
                windows=tot_w, anomalous_windows=anom_w, alerts=alerts,
                tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, f1=f1, pr_auc=pr, roc_auc=roc)


# =====================================================================
# API
# =====================================================================
def auditar(run_dir, data_path=None, verbose=True):
    config, model, meta, thresholds = _load_run(run_dir)
    df_raw, path_used = _resolve_test_df(config, run_dir, data_path)

    target = config.get("data", {}).get("target_col", "Is_Anomaly")
    if target not in df_raw.columns:
        df_raw[target] = 0
    total_events = len(df_raw)
    total_anomalies = int(df_raw[target].sum())

    processor = meta.get("feature_processor")
    temporal = meta.get("temporal_builder")
    if processor is None:
        if verbose:
            print("⚠️  feature_processor ausente — refitando.")
        processor = FeatureProcessor(config)
        processor.fit(df_raw)
    if temporal is None:
        temporal = TemporalBuilder(config)

    t_df = temporal.transform(processor.transform(df_raw))
    _, _, _, users, y_true, scores, seq_len = _build_windows(t_df, config, meta, model)
    y_true = y_true.astype(int)
    y_pred = _thresholds_per_user(users, scores, thresholds)

    if verbose:
        print(f"📂 run_dir : {run_dir}")
        print(f"📂 dataset : {path_used}")
    return _print_report(total_events, total_anomalies, seq_len, y_true, y_pred, scores)


# =====================================================================
# BATCH
# =====================================================================
def _find_runs(root):
    runs = []
    for dirpath, _, filenames in os.walk(root):
        if "config_used.yaml" in filenames and "trainer_meta.pkl" in filenames:
            runs.append(dirpath)
    return sorted(runs)


def auditar_batch(results_root, data_path=None, output_csv=None):
    runs = _find_runs(results_root)
    print(f"🔎 Encontrei {len(runs)} run_dirs em {results_root}", flush=True)
    for r in runs:
        print(f"   • {os.path.relpath(r, results_root)}", flush=True)
    if not runs:
        print("⚠️  Nenhum run_dir encontrado.", flush=True)
        return []

    rows = []
    for i, run_dir in enumerate(runs, 1):
        print(f"\n### [{i}/{len(runs)}] {os.path.relpath(run_dir, results_root)} ###", flush=True)
        try:
            r = auditar(run_dir, data_path=data_path, verbose=False)
            r["run_dir"] = os.path.relpath(run_dir, results_root)
            rows.append(r)
        except Exception as e:
            print(f"   ❌ {type(e).__name__}: {e}", flush=True)

    if output_csv and rows:
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"\n💾 {len(rows)} linhas salvas em: {output_csv}", flush=True)
    return rows


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir")
    p.add_argument("--results_root")
    p.add_argument("--data", default=None)
    p.add_argument("--output_csv", default=None)
    a = p.parse_args()

    if a.results_root:
        auditar_batch(a.results_root, a.data, a.output_csv)

    elif a.run_dir:
        res = auditar(a.run_dir, a.data)
        out_path = a.output_csv or os.path.join(a.run_dir, "auditoria_detecao.csv")
        pd.DataFrame([{**res, "run_dir": a.run_dir}]).to_csv(out_path, index=False)
        print(f"\n💾 CSV salvo em: {out_path}")

    else:
        p.error("Use --run_dir + --data OU --results_root.")