import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from core.data.dataset_adapter import DatasetAdapter
from core.features.feature_processor import FeatureProcessor

config = {
    "sampling": {"mode": "full"},
    "data": {"dataset_path": "data/ef_amplo.jsonl", "target_col": "Is_Anomaly"},
    "features": {"enabled": True}
}

print("1. Testando DatasetAdapter...")
adapter = DatasetAdapter(config)
df = adapter._load_resource("data/ef_amplo.jsonl")
print(f"   -> Carregado: {df.shape[0]} eventos, {df.shape[1]} colunas.")

print("2. Testando FeatureProcessor (fit + transform)...")
processor = FeatureProcessor(config, strict=False)
processor.fit(df)
df_transformed = processor.transform(df)

print("\n3. Sucesso! Verificação do vetor comportamental:")
print(f"   -> Eventos transformados: {df_transformed.shape[0]}")
print(f"   -> Dimensões de features: {len(processor.final_feature_names)}")
print(f"   -> Lista de features: {processor.final_feature_names}")
print(f"   -> Anomalias preservadas: {(df_transformed['Is_Anomaly'] == 1).sum()}")
