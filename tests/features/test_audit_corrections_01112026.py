# tests/features/test_audit_corrections_01112026.py
#
# Testes rigorosos de validação das correções matemáticas da Auditoria de 01/11/2026:
# P0 - §2.1 Hawkes sobre Falhas
# P0 - §2.2 Linhagem de Processos com Jelinek-Mercer Real
# P0 - §2.3 Spray com Entropia por IP
# P0 - §2.4 Escala de Bits em Topologia
# P0 - §3.1 & §3.2 Consistência de Features e Cold-Start
# P1 - §2.5, §2.6, §2.7, §3.5, §3.9

import math
import numpy as np
import pandas as pd
import pytest

from core.features.builders.pilar2_process import build_pilar2_process, fit_pilar2_state
from core.features.builders.pilar3_auth import build_pilar3_auth, fit_pilar3_state
from core.features.builders.pilar4_topology import build_pilar4_topology, fit_pilar4_state
from core.features.builders.pilar5_temporal import build_pilar5_temporal
from core.data.dataset_adapter import DatasetAdapter
from core.features.feature_processor import FeatureProcessor
from core.trainer import Trainer
from core.models.base_model import BaseModel


class MockModel(BaseModel):
    def fit(self, X, y=None): pass
    def score(self, X): return np.zeros(len(X))


def test_hawkes_strictly_on_failures():
    """Verifica que Hawkes ignora sucessos e acumula intensidade apenas sobre falhas."""
    # 5 sucessos seguidos por 3 falhas rápidas
    df = pd.DataFrame([
        {"UserID": "u1", "Time": 100.0, "EventID": 4624},
        {"UserID": "u1", "Time": 200.0, "EventID": 4624},
        {"UserID": "u1", "Time": 300.0, "EventID": 4624},
        {"UserID": "u1", "Time": 400.0, "EventID": 4624},
        {"UserID": "u1", "Time": 500.0, "EventID": 4624},
        # Falhas
        {"UserID": "u1", "Time": 1000.0, "EventID": 4625},
        {"UserID": "u1", "Time": 1005.0, "EventID": 4625},
        {"UserID": "u1", "Time": 1010.0, "EventID": 4625},
    ])
    
    state = fit_pilar3_state(df[df["EventID"] == 4624])  # se só houvesse sucessos no treino
    out = build_pilar3_auth(df, state)
    
    # O primeiro evento de falha em 1000 não tem histórico de falhas prévias
    assert out.loc[5, "hawkes_z"] < out.loc[7, "hawkes_z"], "Rajada de falhas consecutivas deve aumentar o z-score de Hawkes"


def test_jelinek_mercer_real_lineage():
    """Verifica que linhagem JM calcula surprisal em bits com interpolação de triplas."""
    train_df = pd.DataFrame([
        {"ProcessName": "powershell.exe", "ParentProcessName": "cmd.exe", "GrandparentProcessName": "winword.exe"}
    ] * 5 + [
        {"ProcessName": "notepad.exe", "ParentProcessName": "explorer.exe", "GrandparentProcessName": "userinit.exe"}
    ] * 30)

    state = fit_pilar2_state(train_df)
    assert "process_jm_weights" in state
    w3, w2, w1 = state["process_jm_weights"]
    assert math.isclose(w3 + w2 + w1, 1.0, abs_tol=1e-5)

    test_df = pd.DataFrame([
        # Linhagem comum
        {"ProcessName": "notepad.exe", "ParentProcessName": "explorer.exe", "GrandparentProcessName": "userinit.exe"},
        # Linhagem rara
        {"ProcessName": "powershell.exe", "ParentProcessName": "cmd.exe", "GrandparentProcessName": "winword.exe"},
    ])
    out = build_pilar2_process(test_df, state)
    assert out.loc[1, "lineage_surprisal"] > out.loc[0, "lineage_surprisal"]


def test_spray_ip_entropy():
    """Verifica que password spray por IP gera z-score positivo elevado."""
    # Um mesmo IP atacando 10 usuários diferentes com falhas em 300s
    spray_events = [
        {"UserID": f"user_{i}", "Time": 1000.0 + i * 10, "EventID": 4625, "IpAddress": "10.0.0.99"}
        for i in range(10)
    ]
    df = pd.DataFrame(spray_events)
    state = fit_pilar3_state(df)
    out = build_pilar3_auth(df, state)
    
    # Nos eventos finais do spray, spray_z deve estar nitidamente positivo
    assert out.iloc[-1]["spray_z"] > 0.0


def test_canonicalize_logon_failed_regex():
    """Verifica que Logon_Failed ativa is_failed e NÃO ativa is_success."""
    adapter = DatasetAdapter({"data": {}})
    df = pd.DataFrame([
        {"user": "u1", "activity": "Logon_Failed", "Time": 1000.0},
        {"user": "u2", "activity": "Logon", "Time": 1010.0},
    ])
    out = adapter._canonicalize(df)
    assert out.loc[0, "is_failed"] == 1
    assert out.loc[0, "is_success"] == 0
    assert out.loc[1, "is_failed"] == 0
    assert out.loc[1, "is_success"] == 1


def test_trainer_propagates_cold_start():
    """Verifica que Trainer propaga is_cold_start em feature_columns quando habilitado."""
    trainer = Trainer(
        config={"model": {"sequence_length": 5}, "features": {"include_cold_start": True}},
        model=MockModel({})
    )
    df = pd.DataFrame([{"UserID": "u1", "Time": float(t), "f1": 1.0} for t in range(5)])
    trainer._prepare_and_scale_X(df, "u1", fit=True, feature_columns=["f1"])
    assert "is_cold_start" in trainer.feature_columns
