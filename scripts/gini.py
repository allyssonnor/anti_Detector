import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

# Adiciona o caminho do projeto ao sys.path (se ainda não estiver)
project_path = "data/output_thesis_marathon/synthetic/test/test.jsonl"  # ajuste se for outro
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Importa as classes do seu detector
from core.data.dataset_adapter import DatasetAdapter
from core.features.feature_processor import FeatureProcessor

# ----------------------------------------------------------
# 1. CARREGAR OS DADOS BRUTOS USANDO O DATASET ADAPTER
# ----------------------------------------------------------
# Cria uma configuração mínima para o adapter
config = {
    "data": {
        "train_path": "data/output_thesis_marathon/synthetic/train/train.jsonl",
        "val_path": "data/output_thesis_marathon/synthetic/val/val.jsonl",
        "test_path": "data/output_thesis_marathon/synthetic/test/test.jsonl",
        "target_col": "Is_Anomaly",
        "time_nan_strategy": "nan"
    },
    "sampling": {"mode": "full"},
    "features": {"enabled": True}
}

adapter = DatasetAdapter(config)
_, _, test_df = adapter.load()  # só precisamos do test_df

print(f"✅ Dados brutos carregados: {len(test_df)} eventos")

# ----------------------------------------------------------
# 2. PROCESSAR AS FEATURES (gera as 14 colunas)
# ----------------------------------------------------------
# Usa uma instância do FeatureProcessor com o mesmo config
processor = FeatureProcessor(config)

# O fit() precisa ser chamado sobre os dados de treino para aprender
# as distribuições históricas (LogHost_enc e ProcessName_enc)
# Vamos carregar o treino rapidamente para isso
train_df, _, _ = adapter.load()  # recarrega tudo; você pode otimizar depois
processor.fit(train_df)

# Aplica a transformação nos dados de teste
test_df_processed = processor.transform(test_df)

# Agora sim, as colunas de features existem
print("🔍 Colunas disponíveis após o FeatureProcessor:")
print(test_df_processed.columns.tolist())

# ----------------------------------------------------------
# 3. EXTRAIR AS 14 FEATURES E O ALVO
# ----------------------------------------------------------
features = [
    'hour_sin', 'hour_cos', 'hour_dev', 'out_of_hours',
    'event_z', 'fail_rate', 'rdp_rate', 'golden_ticket_score',
    'rare_process', 'rare_process_rate', 'admin_rate',
    'unique_hosts', 'LogHost_enc', 'ProcessName_enc'
]

target = 'Is_Anomaly'

# Verifica se todas as colunas existem
missing = [f for f in features if f not in test_df_processed.columns]
if missing:
    print(f"⚠️ Colunas faltando: {missing}")
    # Se alguma coluna estiver faltando, você pode preencher com 0
    for f in missing:
        test_df_processed[f] = 0

X = test_df_processed[features]
y = test_df_processed[target]

print(f"✅ Dados prontos: {len(X)} amostras, {X.shape[1]} features")

# ----------------------------------------------------------
# 4. MATRIZ DE CORRELAÇÃO (PEARSON)
# ----------------------------------------------------------
print("\n📊 Calculando Matriz de Correlação...")
plt.figure(figsize=(12, 10))
corr_matrix = X.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Matriz de Correlação de Pearson - Atributos Comportamentais")
plt.tight_layout()
plt.savefig("pearson_correlation.png", dpi=300)
print("✅ Gráfico 'pearson_correlation.png' salvo com sucesso!")
plt.show()

# ----------------------------------------------------------
# 5. IMPORTÂNCIA (GINI) COM RANDOM FOREST
# ----------------------------------------------------------
print("\n🌲 Treinando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Importância dos Atributos (Impureza de Gini)")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='steelblue')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
plt.ylabel("Escore de Importância Média")
plt.tight_layout()
plt.savefig("gini_importance.png", dpi=300)
print("✅ Gráfico 'gini_importance.png' salvo com sucesso!")
plt.show()

# Exibe o ranking
print("\n📊 Ranking de Importância (Gini):")
for i in indices:
    print(f"  {features[i]}: {importances[i]:.4f}")

