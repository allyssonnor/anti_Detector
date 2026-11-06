# core/features/feature_processor.py
#
# Feature engineering orchestrator para detecção comportamental UEBA.
# Suporta os presets: E0-legacy, FE-Core, FE-Advanced, FE-Full, FE-Full-No-Deterministic.
# Correções aplicadas (Auditoria 01/11/2026):
# - Descarte rigoroso de features indisponíveis do vetor final (active_specs vs unavailable_features)
# - Suporte a specs multidimensionais (entity_type, time_harmonics, weekday_harmonics, cmd_regex_evasion)
# - Fit seletivo apenas para pilares ativos no preset (otimização de tempo FE-Core / ablações)

import os
import pickle
import logging
from typing import Dict, Any, List, Optional, Set

import numpy as np
import pandas as pd

from core.features.state import FeatureState, compute_param_hash
from core.features.spec import FeatureSpec, FeatureMetadata
from core.features.sets import (
    PRESET_E0_LEGACY,
    PRESET_FE_CORE,
    PRESET_FE_ADVANCED,
    PRESET_FE_FULL,
    PRESET_FE_FULL_NO_DETERMINISTIC,
    normalize_preset_name,
)
from core.features.registry import get_specs_for_preset, FEATURE_REGISTRY
import core.features.builders.legacy as legacy
from core.features.builders.pilar1_identity import build_pilar1_identity
from core.features.builders.pilar2_process import build_pilar2_process, fit_pilar2_state
from core.features.builders.pilar3_auth import build_pilar3_auth, fit_pilar3_state
from core.features.builders.pilar4_topology import build_pilar4_topology, fit_pilar4_state
from core.features.builders.pilar5_temporal import build_pilar5_temporal, fit_pilar5_state
from core.features.builders.pilar6_forensic import build_pilar6_forensic


def _columns_for_spec(spec_name: str) -> List[str]:
    """Mapeia o nome da FeatureSpec para os nomes das colunas físicas geradas no DataFrame."""
    if spec_name == "entity_type":
        return ["entity_type_user", "entity_type_machine", "entity_type_service", "entity_type_system"]
    if spec_name == "time_harmonics":
        return ["time_sin_k1", "time_cos_k1", "time_sin_k2", "time_cos_k2", "time_sin_k3", "time_cos_k3"]
    if spec_name == "weekday_harmonics":
        return ["weekday_sin", "weekday_cos"]
    if spec_name == "cmd_regex_evasion":
        return [f"cmd_evasion_flag_{i}" for i in range(1, 11)]
    return [spec_name]


class FeatureProcessor:
    """
    Orquestrador causal do pipeline de features comportamentais UEBA.
    
    Princípios:
    - Anti-leakage: fit() utiliza apenas eventos normais (Is_Anomaly == 0).
    - Imutabilidade em inferência: transform() nunca altera o estado interno.
    - Versionamento determinístico: snapshot / restore via FeatureState.
    - Preservação estrita: preset 'E0-legacy' reproduz o baseline anterior bit-a-bit.
    """

    def __init__(self, config: Dict[str, Any], strict: bool = True, feature_set: str = PRESET_E0_LEGACY):
        self.config: Dict[str, Any] = config or {}
        self.strict: bool = strict

        feature_cfg = self.config.get("features", {})
        raw_set = feature_cfg.get("feature_set", feature_set)
        self.feature_set: str = normalize_preset_name(raw_set)
        
        self.rate_window: int = max(1, int(feature_cfg.get("rate_window", 20)))
        self.excluded_features: Set[str] = set(feature_cfg.get("exclude_features", []))
        self.output_dir: Optional[str] = self.config.get("output_dir")

        self.features_frozen: bool = False
        self.final_feature_names: List[str] = []
        self.active_builders: List[str] = []
        self.unavailable_features: List[str] = []
        self.is_fitted: bool = False

        self.state: FeatureState = FeatureState(
            feature_version=str(feature_cfg.get("feature_version", "1.0.0"))
        )

    # -----------------------------------------------------------------
    # Retrocompatibilidade com acessos diretos a self.historical_state
    # -----------------------------------------------------------------
    @property
    def historical_state(self) -> Dict[str, Any]:
        return self.state.params

    @historical_state.setter
    def historical_state(self, val: Dict[str, Any]) -> None:
        self.state.params = val if val is not None else {}
        self.state.param_hash = compute_param_hash(self.state.params)

    # -----------------------------------------------------------------
    # Persistência de Estado (Legada e Nova)
    # -----------------------------------------------------------------
    def save_state(self) -> Dict[str, Any]:
        """Salva dicionário legado de auditoria."""
        return {
            "historical_state": self.state.params,
            "active_builders": list(self.active_builders),
            "final_feature_names": list(self.final_feature_names),
            "features_frozen": bool(self.features_frozen),
            "rate_window": int(self.rate_window),
            "is_fitted": bool(self.is_fitted),
            "feature_set": self.feature_set,
            "param_hash": self.state.param_hash,
        }

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Carrega dicionário legado de auditoria."""
        self.state.params = state_dict.get("historical_state", {})
        self.state.param_hash = state_dict.get("param_hash", compute_param_hash(self.state.params))
        self.active_builders = list(state_dict.get("active_builders", []))
        self.final_feature_names = list(state_dict.get("final_feature_names", []))
        self.features_frozen = bool(state_dict.get("features_frozen", False))
        self.rate_window = int(state_dict.get("rate_window", self.rate_window))
        self.is_fitted = bool(state_dict.get("is_fitted", True))
        self.feature_set = state_dict.get("feature_set", self.feature_set)

    def snapshot(self) -> bytes:
        """Serializa o FeatureState completo em bytes."""
        return self.state.serialize()

    @classmethod
    def restore(cls, config: Dict[str, Any], blob: bytes) -> "FeatureProcessor":
        """Restaura uma instância a partir de um snapshot binário."""
        inst = cls(config=config, strict=True)
        inst.state = FeatureState.deserialize(blob)
        inst.is_fitted = True
        return inst

    def _rolling_rate(self, df: pd.DataFrame, source_col: str) -> pd.Series:
        return legacy.rolling_rate(df, source_col, self.rate_window)

    def _time_window_count(self, df: pd.DataFrame, event_col: str, window_seconds: float, exclusive: bool = False) -> pd.Series:
        return legacy.time_window_count(df, event_col, window_seconds, exclusive=exclusive)

    # -----------------------------------------------------------------
    # FIT (Apenas Normais → Anti-Leakage Estrito)
    # -----------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        target_col = self.config.get("data", {}).get("target_col", "Is_Anomaly")
        if target_col in df.columns:
            df_temp = df[df[target_col] == 0].copy()
            if df_temp.empty:
                if self.strict:
                    raise RuntimeError(
                        "FeatureProcessor.fit: nenhum evento 'Is_Anomaly==0' no treino. "
                        "Recusando ajustar baseline sobre eventos anômalos. "
                        "Verifique o split ou rode com strict=False explicitamente."
                    )
                logging.warning("FeatureProcessor.fit: nenhum evento normal. Usando base total.")
                df_temp = df.copy()
        else:
            df_temp = df.copy()

        logging.info("[FeatureProcessor] Ajustando baseline histórico (APENAS NORMAIS)...")

        # 1. Baseline legado de encoders e temporais
        params = legacy.fit_legacy_state(df_temp)

        # 2. Baselines dos novos pilares (apenas pilares ativos no preset)
        if self.feature_set != PRESET_E0_LEGACY:
            specs = get_specs_for_preset(self.feature_set)
            active_pillars = {s.pillar for s in specs}

            if 2 in active_pillars:
                p2 = fit_pilar2_state(df_temp)
                params.update(p2)

            if 3 in active_pillars:
                p3 = fit_pilar3_state(df_temp)
                params.update(p3)

            if 4 in active_pillars:
                p4 = fit_pilar4_state(df_temp)
                for k in ["seen_user_host", "seen_proc_host", "seen_user_ip", "seen_user_logon"]:
                    if k in p4:
                        self.state.seen_sets[k] = p4.pop(k)
                params.update(p4)

            if 5 in active_pillars:
                p5 = fit_pilar5_state(df_temp)
                params.update(p5)

        self.state.params = params
        self.state.param_hash = compute_param_hash(params)
        self.is_fitted = True
        logging.info("Baseline comportamental estabelecido com sucesso.")

    # -----------------------------------------------------------------
    # TRANSFORM (Puro e Causal)
    # -----------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        if not self.is_fitted:
            if getattr(self, "strict", True):
                raise RuntimeError(
                    "FeatureProcessor.transform() chamado sem .fit() prévio. "
                    "Em inferência, carregue o estado do treino via meta['feature_processor']. "
                    "Em treino, chame .fit(train_df) antes de .transform()."
                )
            logging.warning("FeatureProcessor.transform sem .fit prévio — modo permissivo.")
            self.fit(df)

        df = df.copy()
        df = df.sort_values(["UserID", "Time"], kind="stable").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["Time"], unit="s")

        available_cols = set(df.columns)

        # -------------------------------------------------------------
        # Caminho 1: Preset E0-legacy
        # -------------------------------------------------------------
        if self.feature_set == PRESET_E0_LEGACY:
            df = legacy.apply_frequency_encoders(df, self.state.params)

            if not self.features_frozen:
                enabled_groups = set(self.config.get("features", {}).get("enabled_groups", [])) or set(FEATURE_REGISTRY.keys())
                self.active_builders = []
                for name, spec in FEATURE_REGISTRY.items():
                    if name in enabled_groups and set(spec["required"]).issubset(available_cols):
                        self.active_builders.append(spec["builder"])
                self.features_frozen = True

            builder_map = {
                "build_temporal": lambda d: legacy.build_temporal(d, self.state.params),
                "build_auth_failure": lambda d: legacy.build_auth_failure(d, self.rate_window),
                "build_lateral": lambda d: legacy.build_lateral(d, self.state.params, self.rate_window),
                "build_volume": lambda d: legacy.build_volume(d, self.state.params),
                "build_rdp": lambda d: legacy.build_rdp(d, self.rate_window),
                "build_golden_ticket": lambda d: legacy.build_golden_ticket(d),
                "build_process": lambda d: legacy.build_process(d, self.rate_window),
                "build_privilege": lambda d: legacy.build_privilege(d, self.rate_window),
            }
            for b in self.active_builders:
                if b in builder_map:
                    df = builder_map[b](df)

            df = legacy.add_interactions(df)

        # -------------------------------------------------------------
        # Caminho 2: Presets Modulares (FE-Core, FE-Advanced, FE-Full, etc.)
        # -------------------------------------------------------------
        else:
            specs = get_specs_for_preset(self.feature_set)
            active_specs = [s for s in specs if s.is_active(available_cols)]
            self.unavailable_features = [s.name for s in specs if not s.is_active(available_cols)]

            # Prepara state com os sets
            state_params = dict(self.state.params)
            state_params.update(self.state.seen_sets)

            # Executa builders dos pilares selecionados
            active_pillars = {s.pillar for s in active_specs}

            if 1 in active_pillars:
                df = build_pilar1_identity(df, state_params)
            if 2 in active_pillars and "ProcessName" in df.columns:
                df = build_pilar2_process(df, state_params)
            if 3 in active_pillars:
                df = build_pilar3_auth(df, state_params)
            if 4 in active_pillars:
                df = build_pilar4_topology(df, state_params)
            if 5 in active_pillars:
                df = build_pilar5_temporal(df, state_params)
            if 6 in active_pillars:
                df = build_pilar6_forensic(df, state_params)

            # Higienização: descarta colunas pertencentes a specs indisponíveis
            for unavail_spec in self.unavailable_features:
                for col in _columns_for_spec(unavail_spec):
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True, errors="ignore")

        # -------------------------------------------------------------
        # Seleção Final do Vetor Numérico
        # -------------------------------------------------------------
        if not self.final_feature_names:
            if self.feature_set == PRESET_E0_LEGACY:
                self.final_feature_names = [
                    c for c in df.columns
                    if c not in legacy.META_COLS
                    and c not in self.excluded_features
                    and pd.api.types.is_numeric_dtype(df[c])
                ]
            else:
                specs = get_specs_for_preset(self.feature_set)
                active_specs = [s for s in specs if s.is_active(available_cols)]
                candidate_cols: List[str] = []
                for s in active_specs:
                    candidate_cols.extend(_columns_for_spec(s.name))

                self.final_feature_names = [
                    c for c in candidate_cols
                    if c in df.columns
                    and c not in self.excluded_features
                    and pd.api.types.is_numeric_dtype(df[c])
                ]
                self.final_feature_names = list(dict.fromkeys(self.final_feature_names))

            self.features_frozen = True
            logging.info(f"Vetor de features CONGELADO: {len(self.final_feature_names)} dimensões.")
        else:
            for c in self.final_feature_names:
                if c not in df.columns:
                    df[c] = 0.0

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            meta = FeatureMetadata(
                active_features=list(self.final_feature_names),
                unavailable_features=list(self.unavailable_features),
                feature_set=self.feature_set,
                feature_version=self.state.feature_version,
                param_hash=self.state.param_hash,
                fitted_at=self.state.fitted_at,
            )
            with open(os.path.join(self.output_dir, "feature_metadata.pkl"), "wb") as f:
                pickle.dump(meta.to_dict(), f)

        keep_cols = self.final_feature_names + ["UserID", "Time", "Is_Anomaly", "AnomalyType"]
        keep_cols = list(dict.fromkeys(keep_cols))
        return df[[c for c in keep_cols if c in df.columns]].fillna(0)

    # -----------------------------------------------------------------
    # UPDATE (Online)
    # -----------------------------------------------------------------
    def update(self, event: Dict[str, Any]) -> None:
        """Atualiza o estado causal S_{t-} no modo online."""
        if not event or "Time" not in event:
            return
        t_event = float(event["Time"])
        user = str(event.get("UserID", ""))
        host = str(event.get("LogHost", ""))
        if user and host and "seen_user_host" in self.state.seen_sets:
            self.state.seen_sets["seen_user_host"].add(f"{user}@{host}", t_event)
