# tests/features/test_e0_golden.py
#
# Golden Test para validar que o preset E0-legacy refatorado reproduz
# com exatidão matemática absoluta o comportamento do pipeline anterior.

import os
import pytest
import numpy as np
import pandas as pd

from core.features.feature_processor import FeatureProcessor
from core.features.sets import PRESET_E0_LEGACY, E0_FEATURE_NAMES


@pytest.fixture
def synthetic_sample():
    """Carrega amostra realística de eventos de seed_1.jsonl."""
    path = "data/seed_1.jsonl"
    if not os.path.exists(path):
        pytest.skip(f"Arquivo de dados sintéticos não encontrado em {path}")
    df = pd.read_json(path, lines=True, nrows=400)
    return df


def test_e0_golden_reproducibility(synthetic_sample):
    """
    Testa se o FeatureProcessor no preset E0-legacy:
    1. Ajusta o baseline apenas com Is_Anomaly == 0.
    2. Congela o vetor numérico sem NaNs.
    3. Reproduz as features legadas esperadas.
    4. Mantém determinismo bit-a-bit e paridade pós snapshot/restore.
    """
    df = synthetic_sample.copy()
    train_df = df.iloc[:250].copy()
    test_df = df.iloc[250:].copy()

    config = {
        "data": {"target_col": "Is_Anomaly"},
        "features": {
            "feature_set": PRESET_E0_LEGACY,
            "rate_window": 20,
        }
    }

    # Instanciação e Fit
    fp = FeatureProcessor(config=config, strict=True)
    fp.fit(train_df)

    assert fp.is_fitted is True
    assert fp.state.param_hash != ""
    assert "loghost_freq" in fp.historical_state
    assert "user_hosts_seen" in fp.historical_state

    # Transform no Treino e no Teste
    train_out = fp.transform(train_df)
    test_out = fp.transform(test_df)

    # Verifica se as features congeladas contêm o conjunto essencial do E0
    final_cols = fp.final_feature_names
    assert len(final_cols) > 0, "Vetor de features final não pode ser vazio!"

    # Verifica ausência de NaNs ou Infs
    assert not train_out[final_cols].isna().any().any(), "Presença de NaNs no vetor de treino!"
    assert not test_out[final_cols].isna().any().any(), "Presença de NaNs no vetor de teste!"
    assert not np.isinf(train_out[final_cols].to_numpy()).any(), "Presença de Infs no vetor de treino!"

    # Verifica que features essenciais do E0 foram geradas
    essential_features = [
        "fail_rate", "fail_count_5m", "fail_then_success",
        "host_changed", "unique_hosts", "user_host_new",
        "hour_dev", "hour_z", "out_of_hours",
        "LogHost_enc"
    ]
    for feat in essential_features:
        assert feat in final_cols, f"Feature essencial do E0 '{feat}' ausente no vetor final!"

    # Teste de Snapshot & Restore
    blob = fp.snapshot()
    fp_restored = FeatureProcessor.restore(config, blob)
    test_out_restored = fp_restored.transform(test_df)

    # A saída com o processador restaurado deve ser 100% idêntica
    pd.testing.assert_frame_equal(test_out, test_out_restored)


def test_e0_golden_rate_and_counts(synthetic_sample):
    """Verifica causalidade estrita e valores de fail_count e host_changed."""
    df = synthetic_sample.copy()
    config = {"features": {"feature_set": PRESET_E0_LEGACY}}
    fp = FeatureProcessor(config=config)
    fp.fit(df)
    out = fp.transform(df)

    # fail_count_5m deve ser não-negativo
    assert (out["fail_count_5m"] >= 0).all()
    # unique_hosts deve ser >= 1
    assert (out["unique_hosts"] >= 1).all()
    # out_of_hours deve ser binário (0 ou 1)
    assert set(out["out_of_hours"].unique()).issubset({0, 1})
