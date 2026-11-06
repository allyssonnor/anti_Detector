# tests/test_trainer_phase4.py
#
# Testes unitários para as extensões do Trainer na Fase 4:
# - Classificação de papéis (roles) para priors de cold-start
# - Flag de is_cold_start (1{N_hist < 10})
# - Persistência estendida de FeatureState em trainer_meta.pkl

import os
import pickle
import numpy as np
import pandas as pd
import pytest

from core.trainer import Trainer, TRAINING_PROTOCOL_VERSION
from core.features.feature_processor import FeatureProcessor
from core.models.base_model import BaseModel


class DummyModel(BaseModel):
    def fit(self, X, y=None):
        pass
    def score(self, X):
        return np.zeros(len(X))


def test_trainer_role_inference():
    trainer = Trainer(config={}, model=DummyModel({}))
    
    assert trainer._infer_role("DESKTOP-ABC$") == "Machine Account"
    assert trainer._infer_role("SYSTEM") == "System Account"
    assert trainer._infer_role("admin_joao") == "Domain Admin"
    assert trainer._infer_role("svc_backup") == "Service Account"
    assert trainer._infer_role("allysson.noronha") == "Standard User"


def test_trainer_cold_start_flag():
    trainer = Trainer(config={"model": {"sequence_length": 5}, "features": {"include_cold_start": True}}, model=DummyModel({}))
    
    # 5 eventos para user1 (< 10 -> cold start)
    u1_events = pd.DataFrame([{"UserID": "u1", "Time": t, "val": float(t)} for t in range(5)])
    trainer._prepare_and_scale_X(u1_events, "u1", fit=True)
    assert trainer.feature_columns is not None
    assert "is_cold_start" in trainer.feature_columns


def test_trainer_persist_feature_state(tmp_path):
    fp = FeatureProcessor(config={"features": {"feature_set": "E0-legacy"}})
    train_data = pd.DataFrame([
        {"UserID": "u1", "Time": 100.0, "EventID": 4624, "LogHost": "h1", "Is_Anomaly": 0}
    ])
    fp.fit(train_data)
    
    trainer = Trainer(config={"output_dir": str(tmp_path)}, model=DummyModel({}))
    trainer.feature_processor = fp
    
    meta_path = os.path.join(tmp_path, "trainer_meta.pkl")
    trainer._persist_meta(meta_path)
    
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        
    assert meta["training_protocol"] == TRAINING_PROTOCOL_VERSION
    assert "feature_state" in meta
    assert meta["feature_set"] == "E0-legacy"
    assert "param_hash" in meta
    assert meta["param_hash"] == fp.state.param_hash
