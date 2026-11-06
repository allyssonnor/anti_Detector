#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bateria de Inferência Cruzada (Generalistas vs Especialistas).
Executa os 6 cenários de generalização e gera a tabela comparativa.
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import pandas as pd
from scripts.inferir_usual import rodar_inferencia

CONFIG_PATH = os.path.join(root_path, "config", "demo_colab.yaml")
BASE_RESULTS = os.path.join(root_path, "results", "demo_colab", "full", "run_seed42")

SCENARIOS = [
    # --- BLOCO 1: GENERALISTAS (Treinados no EF_Amplo) ---
    {
        "tipo_modelo": "Generalista (EF_Amplo)",
        "modelo": "EF_Amplo_Chrono_RandomForest_global_20",
        "dataset_alvo": "data/seed_1_sample.jsonl",
        "proposito": "Cenário Amplo Multimodal (Seed 1)",
    },
    {
        "tipo_modelo": "Generalista (EF_Amplo)",
        "modelo": "EF_Amplo_Chrono_RandomForest_global_20",
        "dataset_alvo": "data/ef_bruteforce.jsonl",
        "proposito": "Fidelidade em Bruteforce (EvidenceForge)",
    },
    {
        "tipo_modelo": "Generalista (EF_Amplo)",
        "modelo": "EF_Amplo_Chrono_RandomForest_global_20",
        "dataset_alvo": "data/test_bruteforce_42.jsonl",
        "proposito": "Transferência para Ataque Bruteforce 42",
    },

    # --- BLOCO 2: ESPECIALISTAS (Treinados no EF_Bruteforce) ---
    {
        "tipo_modelo": "Especialista (EF_Bruteforce)",
        "modelo": "EF_Bruteforce_Chrono_RandomForest_global_20",
        "dataset_alvo": "data/test_bruteforce_42.jsonl",
        "proposito": "Alta Detecção em Bruteforce Trivial 42",
    },
    {
        "tipo_modelo": "Especialista (EF_Bruteforce)",
        "modelo": "EF_Bruteforce_Chrono_RandomForest_global_20",
        "dataset_alvo": "data/ef_amplo.jsonl",
        "proposito": "Seletividade: Capturar Bruteforce no Amplo",
    },
    {
        "tipo_modelo": "Especialista (EF_Bruteforce)",
        "modelo": "EF_Bruteforce_Chrono_RandomForest_global_20",
        "dataset_alvo": "data/seed_1_sample.jsonl",
        "proposito": "Seletividade: Capturar Bruteforce no Seed 1",
    },
]


def main():
    print("=" * 80)
    print("🚀 INICIANDO MARATONA DE INFERÊNCIA CRUZADA (GENERALISTAS vs ESPECIALISTAS)")
    print("=" * 80)

    summary_records = []

    for sc in SCENARIOS:
        model_dir = os.path.join(BASE_RESULTS, sc["modelo"])
        data_path = os.path.join(root_path, sc["dataset_alvo"])

        if not os.path.exists(model_dir):
            print(f"⚠️ Modelo não encontrado em: {model_dir}. Pule se ainda não treinou.")
            continue
        if not os.path.exists(data_path):
            print(f"⚠️ Dataset não encontrado: {data_path}. Pulando.")
            continue

        print(f"\n🔍 Executando: {sc['tipo_modelo']} -> {sc['dataset_alvo']}")
        print(f"   🎯 Propósito: {sc['proposito']}")

        try:
            metrics = rodar_inferencia(CONFIG_PATH, data_path, model_dir)
            if metrics:
                record = {
                    "Tipo de Modelo": sc["tipo_modelo"],
                    "Modelo": sc["modelo"],
                    "Dataset Alvo": os.path.basename(sc["dataset_alvo"]),
                    "Propósito": sc["proposito"],
                    "Janelas": metrics.get("n_windows", 0),
                    "Anomalias": metrics.get("n_anomalies_true", 0),
                    "Alertas": metrics.get("n_alerts", 0),
                    "TP": metrics.get("tp", 0),
                    "FP": metrics.get("fp", 0),
                    "FN": metrics.get("fn", 0),
                    "TN": metrics.get("tn", 0),
                    "Precision": round(metrics.get("precision", 0.0), 4),
                    "Recall": round(metrics.get("recall", 0.0), 4),
                    "F1-Score": round(metrics.get("f1_score", 0.0), 4),
                    "PR-AUC": round(metrics.get("pr_auc", 0.0), 4),
                    "ROC-AUC": round(metrics.get("roc_auc", 0.0), 4),
                }
                summary_records.append(record)
        except Exception as e:
            print(f"❌ Erro na execução de {sc['modelo']} sobre {sc['dataset_alvo']}: {e}")

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        out_csv = os.path.join(root_path, "results", "demo_colab", "tabela_inferencia_cruzada.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_summary.to_csv(out_csv, index=False)

        print("\n" + "=" * 80)
        print("📊 TABELA CONSOLIDADA DE INFERÊNCIA CRUZADA")
        print("=" * 80)
        display_cols = ["Tipo de Modelo", "Dataset Alvo", "Propósito", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1-Score", "PR-AUC"]
        print(df_summary[display_cols].to_string(index=False))
        print(f"\n💾 Tabela salva em: {out_csv}")
    else:
        print("⚠️ Nenhum resultado de inferência foi gerado.")


if __name__ == "__main__":
    main()
