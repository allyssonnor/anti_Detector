# tests/test_evaluation_phase6.py
#
# Testes para as extensões de avaliação e métricas operacionais da Fase 6.

import os
import json
import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_performance import run_external_evaluation


def test_evaluation_extended_metrics(tmp_path):
    output_dir = str(tmp_path)
    config = {
        "output_dir": output_dir,
        "data": {"target_col": "Is_Anomaly"},
        "model": {"sequence_length": 5},
    }
    
    # 100 amostras com 20 positivas
    np.random.seed(42)
    y_true = np.zeros(100, dtype=int)
    y_true[:20] = 1
    scores = np.random.uniform(0.1, 0.9, size=100)
    scores[:20] += 0.3  # anômalos com escores maiores
    y_pred = (scores > 0.6).astype(int)
    
    metrics = run_external_evaluation(
        config=config,
        scores=scores,
        indices=np.arange(100),
        y_true=y_true,
        y_pred=y_pred,
        users=np.array(["u1"] * 50 + ["u2"] * 50),
        verbose=False
    )
    
    # Validação das novas métricas
    assert "average_precision" in metrics
    assert metrics["average_precision"] is not None
    assert 0.0 <= metrics["average_precision"] <= 1.0
    
    assert "mcc" in metrics
    assert -1.0 <= metrics["mcc"] <= 1.0
    
    assert "balanced_accuracy" in metrics
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    
    assert "f2_score" in metrics
    assert "precision_at_50" in metrics
    assert "recall_at_50" in metrics
    assert "precision_at_100" in metrics
    assert "alerts_per_day" in metrics
    assert "alerts_per_user" in metrics
    
    # Validação do arquivo JSON persistido
    json_path = os.path.join(output_dir, "classification_report.json")
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["pr_auc"] == metrics["pr_auc"]


def test_evaluation_average_precision_nan_when_no_positives(tmp_path):
    config = {"output_dir": str(tmp_path)}
    y_true = np.zeros(50, dtype=int)  # 0 positivos
    scores = np.linspace(0.1, 0.5, 50)
    
    metrics = run_external_evaluation(
        config=config,
        scores=scores,
        indices=np.arange(50),
        y_true=y_true,
        y_pred=np.zeros(50, dtype=int),
        verbose=False
    )
    
    assert metrics["average_precision"] is None
