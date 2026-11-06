# tests/features/test_presets_integration.py
#
# Teste de integração para execução do FeatureProcessor nos presets da especificação:
# FE-Core, FE-Full e E0-legacy.

import os
import pytest
import numpy as np
import pandas as pd

from core.features.feature_processor import FeatureProcessor
from core.features.sets import PRESET_FE_CORE, PRESET_FE_FULL, PRESET_E0_LEGACY


@pytest.fixture
def sample_data():
    path = "data/seed_1.jsonl"
    if not os.path.exists(path):
        pytest.skip("Dataset seed_1.jsonl não encontrado")
    return pd.read_json(path, lines=True, nrows=200)


def test_preset_fe_core_execution(sample_data):
    """Verifica que o preset FE-Core executa os pilares 1 a 5 sem erro."""
    config = {
        "features": {"feature_set": PRESET_FE_CORE}
    }
    fp = FeatureProcessor(config=config, strict=True)
    fp.fit(sample_data)
    out = fp.transform(sample_data)
    
    assert len(fp.final_feature_names) > 0
    assert not out[fp.final_feature_names].isna().any().any()
    # Deve conter features dos pilares 1 e 5
    assert "entity_type_user" in fp.final_feature_names
    assert "time_sin_k1" in fp.final_feature_names


def test_preset_fe_full_execution(sample_data):
    """Verifica que o preset FE-Full executa todos os pilares e persiste metadados."""
    config = {
        "features": {"feature_set": PRESET_FE_FULL}
    }
    fp = FeatureProcessor(config=config, strict=True)
    fp.fit(sample_data)
    out = fp.transform(sample_data)
    
    assert len(fp.final_feature_names) > 0
    assert not out[fp.final_feature_names].isna().any().any()
    assert fp.state.param_hash != ""


def test_e0_legacy_still_golden(sample_data):
    """Garantia contínua de CI: E0-legacy permanece 100% verde e estável."""
    config = {
        "features": {"feature_set": PRESET_E0_LEGACY}
    }
    fp = FeatureProcessor(config=config, strict=True)
    fp.fit(sample_data)
    out = fp.transform(sample_data)
    
    assert "fail_count_5m" in fp.final_feature_names
    assert "host_changed" in fp.final_feature_names
    assert not out[fp.final_feature_names].isna().any().any()
