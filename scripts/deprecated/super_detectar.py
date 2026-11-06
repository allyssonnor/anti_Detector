#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SUPER DETECTAR - EXECUTOR PRINCIPAL DA MARATONA DE TESE
========================================================

Pipeline:
    YAML
      -> DatasetAdapter
      -> split (chronological OU random)
      -> TemporalBuilder.sessionize()
      -> FeatureProcessor UEBA
      -> cache N1 (eventos processados)
      -> cache N2 (índices/metadados das janelas)
      -> Trainer/model
      -> threshold calibrado na validação
      -> avaliação no teste
      -> resultados auditáveis

CACHE ENXUTO
------------
N1:
    DataFrames processados, em Parquet, separados por:
      dataset + seed + split_strategy + pipeline/feature signature

N2:
    NÃO armazena X_train/X_val/X_test completos.
    Armazena apenas:
      - índices finais das janelas
      - users
      - session_ids
      - labels
      - feature_columns
      - metadata do cache

As janelas são reconstruídas em memória a partir do N1.

IMPORTANTE
----------
- Random é mantido como cenário experimental obrigatório.
- Chronological e random percorrem a mesma pipeline.
- Estatísticas aprendidas pelo FeatureProcessor vêm somente do treino.
- Não removemos eventos anômalos antes da formação das janelas.
- No modo não supervisionado, somente JANELAS normais são usadas para treino.
- A construção das janelas respeita session_id.
- O cache é versionado e invalidado quando a configuração relevante muda.
"""

from __future__ import annotations

import os

# Antes de importar TensorFlow/modelos:
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Se quiser GPU, deixe o ambiente decidir. Não forçamos CPU/GPU aqui.

import argparse
import copy
import gc
import hashlib
import json
import pickle
import shutil
import sys
import time
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import yaml

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

try:
    from core.data.dataset_adapter import DatasetAdapter
    from core.features.feature_processor import FeatureProcessor
    from core.temporal_builder import TemporalBuilder
    from core.models.model_factory import create as create_model
    from core.trainer import Trainer
    from scripts.evaluate_performance import run_external_evaluation
    from core.utils.config_validator import ConfigValidator
except ImportError as exc:
    print(f"❌ Erro de importação: {exc}")
    sys.exit(1)


PIPELINE_VERSION = "2026-09-06-super-detectar-cache-v2"
CACHE_SCHEMA_VERSION = "n1-n2-v2"
DEFAULT_MAX_SEQUENCE = 40


class ExperimentRunner:
    def __init__(self, experiment_config_path: str):
        self.config_path = self._resolve_path(experiment_config_path)
        if not self.config_path or not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {experiment_config_path}"
            )

        print(f"📖 Lendo configuração unificada: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle) or {}

        # Validação antecipada. Random é permitido pelo contrato experimental.
        try:
            ConfigValidator(self.config).validate()
        except Exception as exc:
            raise ValueError(f"❌ Configuração inválida: {exc}") from exc

        settings = self.config.get("settings", {})
        self.seeds = settings.get("seeds", [42])
        self.sampling_mode = self.config.get("sampling", {}).get("mode", "full")

        results_root = settings.get("results_root", "results")
        self.results_root = os.path.abspath(
            os.path.join(ROOT_PATH, results_root)
            if not os.path.isabs(str(results_root))
            else str(results_root)
        )

        cache_root = settings.get(
            "cache_root",
            os.path.join(ROOT_PATH, "data", "GLOBAL_CACHE"),
        )
        self.cache_root = os.path.abspath(
            os.path.join(ROOT_PATH, cache_root)
            if not os.path.isabs(str(cache_root))
            else str(cache_root)
        )

        self.results_log: List[dict] = []

        # Diretório de execução: não mistura debug/full.
        self.base_output_dir = os.path.join(
            self.results_root,
            self.sampling_mode,
        )
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.cache_root, exist_ok=True)

    # ------------------------------------------------------------------
    # CAMINHOS / CONFIG
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return path

        candidates = [
            path,
            os.path.join(ROOT_PATH, path),
            os.path.join("/content/drive/MyDrive/Project_Detector", path),
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)
        return path

    @staticmethod
    def _apply_overrides(config: dict, overrides: Optional[dict]) -> dict:
        if not overrides:
            return config

        for key, value in overrides.items():
            parts = key.split(".")
            node = config
            for part in parts[:-1]:
                if not isinstance(node, dict):
                    raise TypeError(
                        f"Não foi possível aplicar override '{key}': "
                        f"'{part}' não é um dicionário."
                    )
                node = node.setdefault(part, {})
            node[parts[-1]] = value

        return config

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, name: str) -> bool:
        if df is None or df.empty:
            print(f"⚠️ [WARNING] {name} está vazio.")
            return False

        required = {"UserID", "Time"}
        missing = required.difference(df.columns)
        if missing:
            print(f"⚠️ [WARNING] {name} não possui colunas obrigatórias: {sorted(missing)}")
            return False

        return True

    # ------------------------------------------------------------------
    # ASSINATURAS / CACHE
    # ------------------------------------------------------------------
    @staticmethod
    def _stable_json(value) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)

    def _feature_signature(self, config: dict) -> str:
        temporal = config.get("temporal", {})
        features = config.get("features", {})
        data = config.get("data", {})

        payload = {
            "pipeline_version": PIPELINE_VERSION,
            "cache_schema": CACHE_SCHEMA_VERSION,
            "session_timeout": temporal.get("session_timeout", 1800),
            "min_events_session": temporal.get("min_events_session", 2),
            "enabled_groups": features.get("enabled_groups", []),
            "features_enabled": features.get("enabled", True),
            "target_col": data.get("target_col", "Is_Anomaly"),
            "time_nan_strategy": data.get("time_nan_strategy", "nan"),
        }
        digest = hashlib.sha256(
            self._stable_json(payload).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    @staticmethod
    def _file_fingerprint(path: Optional[str]) -> str:
        """
        Fingerprint barato e estável suficiente para cache:
        tamanho + mtime_ns + caminho.
        Não lê o dataset inteiro.
        """
        if not path:
            return "no-path"

        p = Path(path)
        if not p.exists():
            return hashlib.sha256(str(path).encode()).hexdigest()[:16]

        stat = p.stat()
        payload = {
            "path": str(p.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        return hashlib.sha256(
            ExperimentRunner._stable_json(payload).encode()
        ).hexdigest()[:16]

    def _dataset_signature(self, config: dict, ds_name: str, seed: int, split_strategy: str) -> str:
        data_cfg = config.get("data", {})
        dataset_path = data_cfg.get("dataset_path")

        payload = {
            "dataset": ds_name,
            "seed": seed,
            "split_strategy": split_strategy.lower(),
            "dataset_fingerprint": self._file_fingerprint(dataset_path),
            "feature_signature": self._feature_signature(config),
            "split_ratios": config.get("sampling", {}).get("split_ratios", {}),
            "train_size": config.get("sampling", {}).get("train_size"),
            "val_size": config.get("sampling", {}).get("val_size"),
            "test_size": config.get("sampling", {}).get("test_size"),
        }

        return hashlib.sha256(
            self._stable_json(payload).encode()
        ).hexdigest()[:20]

    def _cache_dir(self, config: dict, ds_name: str, seed: int) -> Path:
        split_strategy = str(
            config.get("sampling", {}).get("split_strategy", "chronological")
        ).lower()

        signature = self._dataset_signature(
            config,
            ds_name,
            seed,
            split_strategy,
        )

        return (
            Path(self.cache_root)
            / ds_name
            / f"seed_{seed}"
            / split_strategy
            / signature
        )

    @staticmethod
    def _atomic_write_parquet(df: pd.DataFrame, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, destination)

    @staticmethod
    def _atomic_pickle(obj, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, destination)

    @staticmethod
    def _cache_complete(cache_dir: Path) -> bool:
        required = [
            cache_dir / "train.parquet",
            cache_dir / "val.parquet",
            cache_dir / "test.parquet",
            cache_dir / "n1_meta.pkl",
        ]
        return all(path.exists() for path in required)

    # ------------------------------------------------------------------
    # SESSIONIZAÇÃO / N1
    # ------------------------------------------------------------------
    def _sessionize(
        self,
        temporal: TemporalBuilder,
        df: pd.DataFrame,
        name: str,
    ) -> pd.DataFrame:
        """
        Usa a definição canônica de sessão do TemporalBuilder.
        Compatível com a nova arquitetura em que sessionização vem
        antes da FeatureProcessor.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        if hasattr(temporal, "sessionize"):
            out = temporal.sessionize(df)
        else:
            # Fallback de compatibilidade:
            # transform() da versão nova deve ser session-safe.
            out = temporal.transform(df)

        if out is None or out.empty:
            raise ValueError(f"Sessionização resultou em dataset vazio: {name}")

        if "session_id" not in out.columns:
            raise ValueError(
                f"TemporalBuilder não produziu 'session_id' em {name}. "
                "A nova arquitetura exige uma definição canônica de sessão."
            )

        return out

    def _build_n1(self, config: dict, ds_name: str, seed: int):
        split_strategy = str(
            config.get("sampling", {}).get("split_strategy", "chronological")
        ).lower()
        cache_dir = self._cache_dir(config, ds_name, seed)

        if self._cache_complete(cache_dir):
            print(
                f"\n📦 [CACHE N1] Reutilizando dados processados:\n"
                f"   {cache_dir}"
            )

            with open(cache_dir / "n1_meta.pkl", "rb") as handle:
                meta = pickle.load(handle)

            expected_signature = self._dataset_signature(
                config, ds_name, seed, split_strategy
            )
            if meta.get("dataset_signature") != expected_signature:
                print("   ⚠️ Assinatura antiga detectada. Reprocessando N1.")
            else:
                train_df = pd.read_parquet(cache_dir / "train.parquet")
                val_df = pd.read_parquet(cache_dir / "val.parquet")
                test_df = pd.read_parquet(cache_dir / "test.parquet")

                processor = meta["processor"]
                feature_signature = meta["feature_signature"]

                return (
                    train_df,
                    val_df,
                    test_df,
                    processor,
                    feature_signature,
                    cache_dir,
                    meta,
                )

        print(
            f"\n⚙️ [N1] Construindo cache de eventos/features "
            f"({ds_name} | seed={seed} | split={split_strategy})..."
        )

        adapter = DatasetAdapter(config)
        train_raw, val_raw, test_raw = adapter.load()

        if not self._validate_dataframe(train_raw, "train"):
            raise ValueError("Treino inválido/vazio.")
        if not self._validate_dataframe(val_raw, "validation"):
            raise ValueError("Validação inválida/vazia.")
        if not self._validate_dataframe(test_raw, "test"):
            raise ValueError("Teste inválido/vazio.")

        temporal = TemporalBuilder(config)

        # 1) Sessão primeiro.
        train_sessionized = self._sessionize(temporal, train_raw, "train")
        val_sessionized = self._sessionize(temporal, val_raw, "validation")
        test_sessionized = self._sessionize(temporal, test_raw, "test")

        # 2) UEBA: fit somente no treino.
        processor = FeatureProcessor(config)
        if not hasattr(processor, "fit") or not hasattr(processor, "transform"):
            raise TypeError("FeatureProcessor inválido: fit/transform ausentes.")

        processor.fit(train_sessionized)

        # transform nunca deve aprender estado novo.
        train_features = processor.transform(train_sessionized)
        val_features = processor.transform(val_sessionized)
        test_features = processor.transform(test_sessionized)

        for frame_name, frame in [
            ("train_features", train_features),
            ("val_features", val_features),
            ("test_features", test_features),
        ]:
            if frame is None or frame.empty:
                raise ValueError(f"{frame_name} vazio após FeatureProcessor.")

        # session_id precisa sobreviver ao FeatureProcessor.
        if "session_id" not in train_features.columns:
            # O FeatureProcessor pode ser configurado para preservar metadados;
            # caso não preserve, recuperamos por alinhamento de índice.
            raise ValueError(
                "FeatureProcessor não preservou session_id. "
                "A arquitetura atual exige session_id no N1."
            )

        feature_signature = self._feature_signature(config)
        dataset_signature = self._dataset_signature(
            config, ds_name, seed, split_strategy
        )

        meta = {
            "pipeline_version": PIPELINE_VERSION,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "dataset_signature": dataset_signature,
            "feature_signature": feature_signature,
            "split_strategy": split_strategy,
            "seed": seed,
            "dataset_name": ds_name,
            "processor": processor,
            "feature_columns": list(getattr(processor, "final_feature_names", [])),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "session_timeout": config.get("temporal", {}).get("session_timeout", 1800),
            "min_events_session": config.get("temporal", {}).get("min_events_session", 2),
        }

        cache_dir.mkdir(parents=True, exist_ok=True)

        self._atomic_write_parquet(train_features, cache_dir / "train.parquet")
        self._atomic_write_parquet(val_features, cache_dir / "val.parquet")
        self._atomic_write_parquet(test_features, cache_dir / "test.parquet")
        self._atomic_pickle(meta, cache_dir / "n1_meta.pkl")

        print(f"   ✅ N1 salvo em: {cache_dir}")

        del train_raw, val_raw, test_raw
        del train_sessionized, val_sessionized, test_sessionized
        gc.collect()

        return (
            train_features,
            val_features,
            test_features,
            processor,
            feature_signature,
            cache_dir,
            meta,
        )

    # ------------------------------------------------------------------
    # CONSTRUÇÃO EFICIENTE DE JANELAS
    # ------------------------------------------------------------------
    @staticmethod
    def _feature_columns_from_frame(
        df: pd.DataFrame,
        config: dict,
        preferred: Optional[List[str]] = None,
    ) -> List[str]:
        target = config.get("data", {}).get("target_col", "Is_Anomaly")

        if preferred:
            cols = [
                c for c in preferred
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
            if cols:
                return cols

        excluded = {
            "UserID",
            "Time",
            target,
            "AnomalyType",
            "session_id",
            "datetime",
        }

        return [
            c
            for c in df.columns
            if c not in excluded
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    @staticmethod
    def _safe_group_sort(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["UserID", "session_id", "Time"] if c in df.columns]
        return df.sort_values(cols, kind="mergesort").reset_index(drop=True)

    def _build_window_metadata(
        self,
        df: pd.DataFrame,
        config: dict,
        max_sequence: int,
        feature_columns: Optional[List[str]] = None,
    ) -> dict:
        """
        Gera SOMENTE metadados das janelas.

        Cada janela é identificada por seu índice final no DataFrame N1.
        Como não atravessamos session_id, basta armazenar os finais válidos.

        Para seq <= max_sequence, os mesmos eventos podem ser usados para
        reconstruir a janela sem duplicar X em disco.
        """
        if df is None or df.empty:
            return {
                "end_indices": np.array([], dtype=np.int64),
                "users": np.array([], dtype=str),
                "sessions": np.array([], dtype=str),
                "labels": np.array([], dtype=np.int8),
                "feature_columns": feature_columns or [],
                "max_sequence": max_sequence,
            }

        work = self._safe_group_sort(df)

        target = config.get("data", {}).get("target_col", "Is_Anomaly")
        feat_cols = self._feature_columns_from_frame(
            work,
            config,
            preferred=feature_columns,
        )

        ends: List[int] = []
        users: List[str] = []
        sessions: List[str] = []
        labels: List[int] = []

        if "session_id" in work.columns:
            groups = work.groupby(
                ["UserID", "session_id"],
                sort=False,
                dropna=False,
            )
        else:
            # Não deveria ocorrer na arquitetura nova.
            groups = work.groupby("UserID", sort=False, dropna=False)

        global_positions = np.arange(len(work), dtype=np.int64)

        for group_key, group in groups:
            n = len(group)
            if n < max_sequence:
                continue

            pos = global_positions[group.index.to_numpy()]
            ends.extend(pos[max_sequence - 1 :].tolist())

            user_value = (
                group_key[0] if isinstance(group_key, tuple) else group_key
            )
            session_value = (
                group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1
                else group.get("session_id", pd.Series([""] * n)).iloc[0]
            )

            users.extend([str(user_value)] * (n - max_sequence + 1))
            sessions.extend([str(session_value)] * (n - max_sequence + 1))

            if target in group.columns:
                y = group[target].fillna(0).astype(int).to_numpy()
                # O rótulo da janela é max(eventos da janela).
                from numpy.lib.stride_tricks import sliding_window_view

                yw = sliding_window_view(y, max_sequence)
                labels.extend(np.max(yw, axis=1).astype(np.int8).tolist())
            else:
                labels.extend([0] * (n - max_sequence + 1))

        return {
            "end_indices": np.asarray(ends, dtype=np.int64),
            "users": np.asarray(users, dtype=str),
            "sessions": np.asarray(sessions, dtype=str),
            "labels": np.asarray(labels, dtype=np.int8),
            "feature_columns": feat_cols,
            "max_sequence": int(max_sequence),
        }

    @staticmethod
    def _reconstruct_windows_from_metadata(
        df: pd.DataFrame,
        metadata: dict,
        sequence_length: int,
        drop_normal_sessions_with_short_length: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstrói X a partir do N1 + índices N2.

        A implementação usa cada session_id separadamente.
        O N2 pode ter sido criado para max_sequence >= sequence_length.

        O índice final de uma janela é o endpoint no N1.
        """
        if df is None or df.empty:
            return (
                np.empty((0, sequence_length, len(metadata.get("feature_columns", []))), dtype=np.float32),
                np.array([], dtype=np.int64),
                np.array([], dtype=str),
                np.array([], dtype=np.int8),
            )

        work = ExperimentRunner._safe_group_sort(df)
        feat_cols = metadata["feature_columns"]

        if not feat_cols:
            raise ValueError("N2 não possui feature_columns.")

        if sequence_length > int(metadata["max_sequence"]):
            raise ValueError(
                f"sequence_length={sequence_length} excede max_sequence="
                f"{metadata['max_sequence']} do N2."
            )

        endpoint_set = set(metadata["end_indices"].tolist())

        X_list: List[np.ndarray] = []
        idx_list: List[np.ndarray] = []
        user_list: List[np.ndarray] = []
        label_list: List[np.ndarray] = []

        target = "Is_Anomaly"

        if "session_id" in work.columns:
            groups = work.groupby(
                ["UserID", "session_id"],
                sort=False,
                dropna=False,
            )
        else:
            groups = work.groupby("UserID", sort=False, dropna=False)

        for group_key, group in groups:
            gidx = group.index.to_numpy()
            if len(group) < sequence_length:
                continue

            x = (
                group.reindex(columns=feat_cols, fill_value=0)
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )

            from numpy.lib.stride_tricks import sliding_window_view

            windows = sliding_window_view(
                x,
                window_shape=sequence_length,
                axis=0,
            )
            windows = np.swapaxes(windows, 1, 2)

            # Endpoints relativos ao grupo.
            local_ends = np.arange(
                sequence_length - 1,
                len(group),
                dtype=np.int64,
            )

            global_ends = gidx[local_ends]
            keep_mask = np.array([int(e) in endpoint_set for e in global_ends])

            if not keep_mask.any():
                continue

            windows = windows[keep_mask]
            global_ends = global_ends[keep_mask]

            user_value = (
                group_key[0] if isinstance(group_key, tuple) else group_key
            )

            X_list.append(windows)
            idx_list.append(global_ends.astype(np.int64))
            user_list.append(np.full(len(global_ends), str(user_value), dtype=str))

            if target in group.columns:
                y = group[target].fillna(0).astype(int).to_numpy()
                yw = sliding_window_view(y, sequence_length)
                yw = np.max(yw, axis=1)
                # local labels correspond to all possible windows, then mask.
                label_list.append(yw[keep_mask].astype(np.int8))
            else:
                label_list.append(np.zeros(len(global_ends), dtype=np.int8))

        if not X_list:
            return (
                np.empty(
                    (0, sequence_length, len(feat_cols)),
                    dtype=np.float32,
                ),
                np.array([], dtype=np.int64),
                np.array([], dtype=str),
                np.array([], dtype=np.int8),
            )

        return (
            np.concatenate(X_list, axis=0),
            np.concatenate(idx_list, axis=0),
            np.concatenate(user_list, axis=0),
            np.concatenate(label_list, axis=0),
        )

    # ------------------------------------------------------------------
    # CACHE N2
    # ------------------------------------------------------------------
    @staticmethod
    def _n2_dir(cache_dir: Path, max_sequence: int) -> Path:
        return cache_dir / "n2" / f"max_seq_{max_sequence}"

    def _n2_complete(self, n2_dir: Path) -> bool:
        required = [
            n2_dir / "train_meta.npz",
            n2_dir / "val_meta.npz",
            n2_dir / "test_meta.npz",
            n2_dir / "n2_meta.pkl",
        ]
        return all(path.exists() for path in required)

    @staticmethod
    def _save_window_metadata(meta: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            end_indices=meta["end_indices"],
            users=meta["users"],
            sessions=meta["sessions"],
            labels=meta["labels"],
        )

    @staticmethod
    def _load_window_metadata(path: Path, feature_columns: List[str], max_sequence: int) -> dict:
        data = np.load(path, allow_pickle=False)
        return {
            "end_indices": data["end_indices"],
            "users": data["users"],
            "sessions": data["sessions"],
            "labels": data["labels"],
            "feature_columns": feature_columns,
            "max_sequence": max_sequence,
        }

    def _build_or_load_n2(
        self,
        cache_dir: Path,
        config: dict,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: List[str],
        max_sequence: int,
    ):
        n2_dir = self._n2_dir(cache_dir, max_sequence)

        if self._n2_complete(n2_dir):
            with open(n2_dir / "n2_meta.pkl", "rb") as handle:
                n2_meta = pickle.load(handle)

            if (
                n2_meta.get("pipeline_version") == PIPELINE_VERSION
                and n2_meta.get("max_sequence") == max_sequence
                and n2_meta.get("feature_columns") == feature_columns
            ):
                print(f"   ⚡ [CACHE N2] Reutilizando metadados de janelas: {n2_dir}")

                return (
                    self._load_window_metadata(
                        n2_dir / "train_meta.npz",
                        feature_columns,
                        max_sequence,
                    ),
                    self._load_window_metadata(
                        n2_dir / "val_meta.npz",
                        feature_columns,
                        max_sequence,
                    ),
                    self._load_window_metadata(
                        n2_dir / "test_meta.npz",
                        feature_columns,
                        max_sequence,
                    ),
                    n2_meta,
                )

        print(
            f"   ⚙️ [N2] Gerando somente índices/metadados "
            f"(max_seq={max_sequence})..."
        )

        train_meta = self._build_window_metadata(
            train_df,
            config,
            max_sequence,
            feature_columns,
        )
        val_meta = self._build_window_metadata(
            val_df,
            config,
            max_sequence,
            feature_columns,
        )
        test_meta = self._build_window_metadata(
            test_df,
            config,
            max_sequence,
            feature_columns,
        )

        n2_dir.mkdir(parents=True, exist_ok=True)

        self._save_window_metadata(train_meta, n2_dir / "train_meta.npz")
        self._save_window_metadata(val_meta, n2_dir / "val_meta.npz")
        self._save_window_metadata(test_meta, n2_dir / "test_meta.npz")

        n2_meta = {
            "pipeline_version": PIPELINE_VERSION,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "max_sequence": max_sequence,
            "feature_columns": feature_columns,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "counts": {
                "train": int(len(train_meta["end_indices"])),
                "val": int(len(val_meta["end_indices"])),
                "test": int(len(test_meta["end_indices"])),
            },
        }

        self._atomic_pickle(n2_meta, n2_dir / "n2_meta.pkl")

        return train_meta, val_meta, test_meta, n2_meta

    # ------------------------------------------------------------------
    # SCALING
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_per_user_scalers(
        train_df: pd.DataFrame,
        feature_columns: List[str],
        normal_mask: np.ndarray,
    ) -> Dict[str, object]:
        """
        Mantemos o contrato do Trainer: scaler por usuário.
        Para não duplicar lógica de treinamento, usamos StandardScaler.
        """
        from sklearn.preprocessing import StandardScaler

        scalers: Dict[str, StandardScaler] = {}

        train_work = train_df.reset_index(drop=True)
        user_values = train_work["UserID"].astype(str).to_numpy()

        normal_rows = train_work.loc[normal_mask].copy()

        for user_id, group in normal_rows.groupby("UserID", sort=False):
            x = (
                group.reindex(columns=feature_columns, fill_value=0)
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )

            if len(x) == 0:
                continue

            scaler = StandardScaler()
            scaler.fit(x)
            scalers[str(user_id)] = scaler

        return scalers

    @staticmethod
    def _scale_event_frame(
        df: pd.DataFrame,
        feature_columns: List[str],
        scalers: Dict[str, object],
    ) -> pd.DataFrame:
        work = df.copy()

        # O scaler sempre produz floats. Algumas features UEBA são binárias
        # ou inteiras no N1; manter o DataFrame como int64 faz o pandas emitir
        # FutureWarning ao receber os valores escalados. Mais importante: em
        # versões futuras isso poderá virar erro. O cast explícito para float32
        # torna o contrato de tipos inequívoco e reduz memória.
        numeric = (
            work.reindex(columns=feature_columns, fill_value=0)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .astype(np.float32)
        )

        out = numeric.copy()

        for user_id, idx in work.groupby("UserID", sort=False).groups.items():
            key = str(user_id)
            rows = numeric.loc[idx].to_numpy(dtype=np.float32)

            scaler = scalers.get(key)
            if scaler is None:
                # Fallback para usuário não visto no treino: sem ajuste aprendido.
                scaled = rows
            else:
                scaled = scaler.transform(rows)

            out.loc[idx, feature_columns] = np.asarray(
                scaled,
                dtype=np.float32,
            )

        result = work.copy()
        for col in feature_columns:
            result[col] = out[col].astype(np.float32)

        return result

    def _prepare_scaled_frames(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: List[str],
        train_n2: dict,
    ):
        target = self.config_target(train_df)
        normal_mask = (
            train_df[target].fillna(0).astype(int).to_numpy() == 0
            if target in train_df.columns
            else np.ones(len(train_df), dtype=bool)
        )

        scalers = self._fit_per_user_scalers(
            train_df,
            feature_columns,
            normal_mask,
        )

        return (
            self._scale_event_frame(train_df, feature_columns, scalers),
            self._scale_event_frame(val_df, feature_columns, scalers),
            self._scale_event_frame(test_df, feature_columns, scalers),
            scalers,
        )

    @staticmethod
    def config_target(df: pd.DataFrame) -> str:
        return "Is_Anomaly"

    # ------------------------------------------------------------------
    # AVALIAÇÃO
    # ------------------------------------------------------------------
    def _evaluate_subset(
        self,
        trainer: Trainer,
        test_df: pd.DataFrame,
        output_dir: str,
        config: dict,
        subset_name: str,
        precomputed_test=None,
        verbose: bool = True,
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.perf_counter()

        if precomputed_test is None:
            raise ValueError(
                "A avaliação oficial deve receber o bundle pré-computado "
                "para manter alinhamento exato entre N1/N2 e o resultado."
            )

        X_test, indices, users, y_test = precomputed_test

        if len(X_test) == 0:
            if verbose:
                print(f"   ⚠️ Nenhuma sequência válida para {subset_name}.")
            return {
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "elapsed_sec": 0.0,
                "mem_peak_mb": 0.0,
            }

        scores = trainer._calculate_scores(X_test)

        global_fallback = trainer.global_threshold
        thresholds_aplicados = np.asarray(
            [
                trainer.user_thresholds.get(str(user), global_fallback)
                for user in users
            ],
            dtype=float,
        )
        y_pred = (scores > thresholds_aplicados).astype(int)

        np.save(os.path.join(output_dir, "scores.npy"), scores)
        np.save(os.path.join(output_dir, "y_pred.npy"), y_pred)
        np.save(os.path.join(output_dir, "indices.npy"), indices)
        np.save(os.path.join(output_dir, "y_true.npy"), y_test)
        np.save(os.path.join(output_dir, "users.npy"), users)

        source_thresholds = os.path.join(
            trainer.output_dir,
            "user_thresholds.json",
        )
        if os.path.exists(source_thresholds):
            shutil.copy2(
                source_thresholds,
                os.path.join(output_dir, "user_thresholds.json"),
            )

        eval_config = copy.deepcopy(config)
        eval_config["output_dir"] = output_dir

        metrics = run_external_evaluation(
            eval_config,
            test_df=test_df,
            scores=scores,
            indices=indices,
            verbose=verbose,
        )

        elapsed = time.perf_counter() - start_time

        try:
            mem_peak_mb = (
                psutil.Process().memory_info().peak_wset / (1024 * 1024)
                if sys.platform == "win32"
                else psutil.Process().memory_info().rss / (1024 * 1024)
            )
        except Exception:
            mem_peak_mb = 0.0

        return {
            **metrics,
            "elapsed_sec": elapsed,
            "mem_peak_mb": mem_peak_mb,
        }

    # ------------------------------------------------------------------
    # EXECUÇÃO
    # ------------------------------------------------------------------
    def run(self):
        datasets = self.config.get("datasets", [])
        model_presets = self.config.get("model_presets", [])
        hparam_grid = self.config.get("hyperparameter_grid", {})

        if not datasets:
            raise ValueError("Nenhum dataset definido em 'datasets'.")
        if not model_presets:
            raise ValueError("Nenhum model_preset definido.")

        grid_keys = list(hparam_grid.keys())
        grid_values = list(hparam_grid.values())
        grid_combinations = (
            list(product(*grid_values))
            if grid_values
            else [()]
        )

        total_runs = (
            len(datasets)
            * len(model_presets)
            * max(1, len(grid_combinations))
            * len(self.seeds)
        )

        print("\n" + "=" * 78)
        print("🚀 MARATONA DE EXPERIMENTOS — SUPER DETECTAR")
        print("=" * 78)
        print(f" Pipeline          : {PIPELINE_VERSION}")
        print(f" Cache              : {self.cache_root}")
        print(f" Modo               : {self.sampling_mode.upper()}")
        print(f" Datasets           : {len(datasets)}")
        print(f" Modelos            : {len(model_presets)}")
        print(f" Grid               : {max(1, len(grid_combinations))}")
        print(f" Seeds              : {len(self.seeds)} {self.seeds}")
        print(f" TOTAL DE RUNS     : {total_runs}")
        print("=" * 78)

        for seed in self.seeds:
            print(
                f"\n{'#' * 78}\n"
                f"🌱 SEED {seed}\n"
                f"{'#' * 78}"
            )

            config_seed = copy.deepcopy(self.config)
            config_seed.setdefault("training", {})["seed"] = seed

            seed_output_dir = os.path.join(
                self.base_output_dir,
                f"run_seed{seed}",
            )
            os.makedirs(seed_output_dir, exist_ok=True)

            self.results_log = []
            master_csv_path = os.path.join(
                seed_output_dir,
                "master_results_table.csv",
            )

            for ds in datasets:
                ds_name = ds.get("name", "unknown_dataset")
                config_ds = self._apply_overrides(
                    copy.deepcopy(config_seed),
                    ds.get("overrides", {}),
                )

                split_strategy = str(
                    config_ds.get("sampling", {}).get(
                        "split_strategy",
                        "chronological",
                    )
                ).lower()

                print(
                    f"\n📚 DATASET: {ds_name} | "
                    f"SPLIT: {split_strategy.upper()} | SEED: {seed}"
                )

                # ------------------------------------------------------
                # N1
                # ------------------------------------------------------
                (
                    train_df,
                    val_df,
                    test_df,
                    processor,
                    feature_signature,
                    cache_dir,
                    n1_meta,
                ) = self._build_n1(
                    config_ds,
                    ds_name,
                    seed,
                )

                feature_columns = self._feature_columns_from_frame(
                    train_df,
                    config_ds,
                    preferred=n1_meta.get("feature_columns"),
                )

                if not feature_columns:
                    raise ValueError(
                        f"Nenhuma feature numérica válida em {ds_name}."
                    )

                # ------------------------------------------------------
                # Definição do MAX_SEQUENCE do cache N2
                # ------------------------------------------------------
                requested_sequences = {
                    int(p.get("overrides", {}).get("model.sequence_length", 0))
                    for p in model_presets
                    if p.get("overrides", {}).get("model.sequence_length") is not None
                }

                grid_sequences = {
                    int(v)
                    for key, values in hparam_grid.items()
                    if key == "model.sequence_length"
                    for v in values
                    if v is not None
                }

                all_sequences = {
                    x for x in requested_sequences.union(grid_sequences) if x > 0
                }

                if not all_sequences:
                    all_sequences = {
                        int(config_ds.get("model", {}).get("sequence_length", 20))
                    }

                max_sequence = max(
                    max(all_sequences),
                    DEFAULT_MAX_SEQUENCE if max(all_sequences) <= DEFAULT_MAX_SEQUENCE and 40 in all_sequences else max(all_sequences),
                )

                # Para a grade da tese 10..40, max_sequence=40.
                if any(x in all_sequences for x in [10, 15, 20, 25, 30, 35, 40]):
                    max_sequence = max(max_sequence, 40)

                (
                    n2_train,
                    n2_val,
                    n2_test,
                    n2_meta,
                ) = self._build_or_load_n2(
                    cache_dir,
                    config_ds,
                    train_df,
                    val_df,
                    test_df,
                    feature_columns,
                    max_sequence,
                )

                # ------------------------------------------------------
                # Escala: somente treino normal
                # ------------------------------------------------------
                from sklearn.preprocessing import StandardScaler

                target = config_ds.get("data", {}).get("target_col", "Is_Anomaly")
                normal_mask = (
                    train_df[target].fillna(0).astype(int).to_numpy() == 0
                    if target in train_df.columns
                    else np.ones(len(train_df), dtype=bool)
                )

                scalers = self._fit_per_user_scalers(
                    train_df,
                    feature_columns,
                    normal_mask,
                )

                train_scaled = self._scale_event_frame(
                    train_df, feature_columns, scalers
                )
                val_scaled = self._scale_event_frame(
                    val_df, feature_columns, scalers
                )
                test_scaled = self._scale_event_frame(
                    test_df, feature_columns, scalers
                )

                # ------------------------------------------------------
                # MODELOS / GRID
                # ------------------------------------------------------
                for mp in model_presets:
                    mp_name = mp.get("name", "unknown_model")

                    for combo in grid_combinations:
                        current_hparams = dict(zip(grid_keys, combo))

                        config_final = self._apply_overrides(
                            copy.deepcopy(config_ds),
                            mp.get("overrides", {}),
                        )
                        for key, value in current_hparams.items():
                            config_final = self._apply_overrides(
                                config_final,
                                {key: value},
                            )

                        seq_len = int(
                            config_final.get("model", {}).get(
                                "sequence_length",
                                10,
                            )
                        )

                        if seq_len > max_sequence:
                            raise ValueError(
                                f"sequence_length={seq_len} > max_sequence={max_sequence}."
                            )

                        is_supervised = bool(
                            config_final.get("training", {}).get(
                                "supervised",
                                False,
                            )
                        )

                        sup_str = (
                            "supervised"
                            if is_supervised
                            else "unsupervised"
                        )

                        print(
                            f"\n   🏋️ {mp_name} | "
                            f"seq={seq_len} | "
                            f"{sup_str.upper()} | "
                            f"split={split_strategy.upper()}"
                        )

                        # CACHE_ONLY é um preset de pré-processamento.
                        # Não deve reconstruir X_train/X_val/X_test nem instanciar
                        # ou treinar modelo: para seq=40 isso pode materializar
                        # centenas de milhares de janelas 3D e consumir vários GB.
                        # O N2 já contém somente os índices/metadados necessários.
                        if mp_name.strip().upper() == "CACHE_ONLY":
                            model_string = (
                                f"{ds_name}_{mp_name}"
                                + (
                                    "_" + "_".join(str(v) for v in combo)
                                    if combo
                                    else ""
                                )
                            ).replace("/", "-").replace("\\", "-")

                            run_dir = os.path.join(seed_output_dir, model_string)
                            os.makedirs(run_dir, exist_ok=True)
                            config_final["output_dir"] = run_dir

                            cache_only_meta = {
                                "pipeline_version": PIPELINE_VERSION,
                                "cache_schema_version": CACHE_SCHEMA_VERSION,
                                "dataset_name": ds_name,
                                "split_strategy": split_strategy,
                                "seed": seed,
                                "mode": "CACHE_ONLY",
                                "status": "cache_ready",
                                "sequence_length": seq_len,
                                "n2_max_sequence": max_sequence,
                                "feature_signature": feature_signature,
                                "feature_columns": feature_columns,
                                "cache_dir": str(cache_dir),
                                "train_events": int(len(train_df)),
                                "val_events": int(len(val_df)),
                                "test_events": int(len(test_df)),
                                "train_windows": int(len(n2_train.get("end_indices", []))),
                                "val_windows": int(len(n2_val.get("end_indices", []))),
                                "test_windows": int(len(n2_test.get("end_indices", []))),
                            }

                            with open(
                                os.path.join(run_dir, "cache_only_meta.json"),
                                "w",
                                encoding="utf-8",
                            ) as handle:
                                json.dump(cache_only_meta, handle, indent=2, ensure_ascii=False)

                            with open(
                                os.path.join(run_dir, "config_used.yaml"),
                                "w",
                                encoding="utf-8",
                            ) as handle:
                                yaml.safe_dump(
                                    config_final,
                                    handle,
                                    sort_keys=False,
                                    allow_unicode=True,
                                )

                            self.results_log.append({
                                "pipeline_version": PIPELINE_VERSION,
                                "cache_schema_version": CACHE_SCHEMA_VERSION,
                                "seed": seed,
                                "dataset": ds_name,
                                "split_strategy": split_strategy,
                                "model_preset": mp_name,
                                "sequence_length": seq_len,
                                "supervised": is_supervised,
                                "status": "cache_ready",
                                "train_events": int(len(train_df)),
                                "val_events": int(len(val_df)),
                                "test_events": int(len(test_df)),
                                "train_windows": int(len(n2_train.get("end_indices", []))),
                                "val_windows": int(len(n2_val.get("end_indices", []))),
                                "test_windows": int(len(n2_test.get("end_indices", []))),
                                **current_hparams,
                            })

                            pd.DataFrame(self.results_log).to_csv(
                                master_csv_path,
                                index=False,
                            )

                            print(
                                "   ✅ CACHE_ONLY concluído: N1/N2 prontos; "
                                "nenhuma janela 3D foi materializada e nenhum modelo foi treinado."
                            )
                            continue

                        # Reconstrói janelas para o seq_len escolhido.
                        X_train_all, train_indices, train_users, y_train_all = (
                            self._reconstruct_windows_from_metadata(
                                train_scaled,
                                n2_train,
                                seq_len,
                            )
                        )
                        X_val, val_indices, val_users, y_val = (
                            self._reconstruct_windows_from_metadata(
                                val_scaled,
                                n2_val,
                                seq_len,
                            )
                        )
                        X_test, test_indices, test_users, y_test = (
                            self._reconstruct_windows_from_metadata(
                                test_scaled,
                                n2_test,
                                seq_len,
                            )
                        )

                        if len(X_train_all) == 0:
                            raise ValueError(
                                f"Treino não gerou janelas: {ds_name}/{mp_name}/seq={seq_len}"
                            )
                        if len(X_val) == 0:
                            raise ValueError(
                                f"Validação não gerou janelas: {ds_name}/{mp_name}/seq={seq_len}"
                            )

                        # CRÍTICO:
                        # construir janelas primeiro; só depois selecionar
                        # as JANELAS normais para o treino não supervisionado.
                        if is_supervised:
                            X_train = X_train_all
                            y_train = y_train_all
                        else:
                            normal_window_mask = (y_train_all == 0)
                            X_train = X_train_all[normal_window_mask]
                            y_train = y_train_all[normal_window_mask]

                        if len(X_train) == 0:
                            raise ValueError(
                                "Nenhuma janela normal disponível para treino "
                                f"não supervisionado: {ds_name}/{mp_name}."
                            )

                        model_string = (
                            f"{ds_name}_{mp_name}"
                            + (
                                "_"
                                + "_".join(str(v) for v in combo)
                                if combo
                                else ""
                            )
                        ).replace("/", "-").replace("\\", "-")

                        run_dir = os.path.join(
                            seed_output_dir,
                            model_string,
                        )
                        os.makedirs(run_dir, exist_ok=True)
                        config_final["output_dir"] = run_dir

                        model = create_model(config_final)
                        trainer = Trainer(config_final, model)

                        # Injeta o estado de escala, já aprendido somente no treino normal.
                        trainer.scalers = scalers
                        trainer.feature_columns = feature_columns

                        # Preserva metadados para inferência/reprodução.
                        optim_metric = config_final.get(
                            "training", {}
                        ).get(
                            "optimization_metric",
                            "f2",
                        )

                        results = trainer.run(
                            precomputed_train=(X_train, y_train),
                            precomputed_val=(
                                X_val,
                                val_users,
                                y_val,
                            ),
                            optimization_metric=optim_metric,
                        )

                        # Salva configuração usada.
                        with open(
                            os.path.join(run_dir, "config_used.yaml"),
                            "w",
                            encoding="utf-8",
                        ) as handle:
                            yaml.safe_dump(
                                config_final,
                                handle,
                                sort_keys=False,
                                allow_unicode=True,
                            )

                        # Enriquece trainer_meta.
                        trainer_meta_path = os.path.join(
                            run_dir,
                            "trainer_meta.pkl",
                        )

                        if os.path.exists(trainer_meta_path):
                            with open(trainer_meta_path, "rb") as handle:
                                trainer_meta = pickle.load(handle)
                        else:
                            trainer_meta = {}

                        trainer_meta.update(
                            {
                                "pipeline_version": PIPELINE_VERSION,
                                "cache_schema_version": CACHE_SCHEMA_VERSION,
                                "dataset_name": ds_name,
                                "split_strategy": split_strategy,
                                "seed": seed,
                                "sequence_length": seq_len,
                                "feature_signature": feature_signature,
                                "feature_columns": feature_columns,
                                "scalers": scalers,
                                "feature_processor": processor,
                                "cache_dir": str(cache_dir),
                                "n2_max_sequence": max_sequence,
                                "temporal_config": config_final.get(
                                    "temporal",
                                    {},
                                ),
                            }
                        )

                        with open(
                            trainer_meta_path,
                            "wb",
                        ) as handle:
                            pickle.dump(
                                trainer_meta,
                                handle,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )

                        # --------------------------------------------------
                        # Avaliação
                        # --------------------------------------------------
                        run_mixed_eval = bool(
                            config_final.get("settings", {}).get(
                                "run_mixed_eval",
                                True,
                            )
                        )

                        if run_mixed_eval:
                            test_bundle = (
                                X_test,
                                test_indices,
                                test_users,
                                y_test,
                            )

                            eval_dir = os.path.join(
                                run_dir,
                                "eval_mixed",
                            )

                            metrics = self._evaluate_subset(
                                trainer,
                                test_scaled,
                                eval_dir,
                                config_final,
                                ds_name,
                                precomputed_test=test_bundle,
                                verbose=True,
                            )

                            self.results_log.append(
                                {
                                    "pipeline_version": PIPELINE_VERSION,
                                    "seed": seed,
                                    "dataset": ds_name,
                                    "split_strategy": split_strategy,
                                    "model_preset": mp_name,
                                    "sequence_length": seq_len,
                                    "supervised": is_supervised,
                                    **current_hparams,
                                    **results,
                                    **metrics,
                                }
                            )

                        pd.DataFrame(self.results_log).to_csv(
                            master_csv_path,
                            index=False,
                        )

                        print(
                            f"   ✅ Finalizado: {model_string}\n"
                        )

                        # Libera matrizes grandes antes da próxima combinação.
                        del (
                            X_train_all,
                            X_train,
                            y_train_all,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                        )
                        gc.collect()

                # Libera N1 do dataset corrente.
                del train_df, val_df, test_df
                del train_scaled, val_scaled, test_scaled
                gc.collect()

        print("\n" + "=" * 78)
        print("✅ MARATONA CONCLUÍDA")
        print("=" * 78)
        print(f"📄 Resultados: {self.base_output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executor principal da maratona UEBA."
    )
    parser.add_argument(
        "--config",
        "--exp_config",
        dest="config",
        required=True,
        help="Caminho do YAML da maratona.",
    )
    args = parser.parse_args()

    runner = ExperimentRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()
