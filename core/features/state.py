# core/features/state.py
#
# Representação serializável e versionada do estado causal S_{t-}.
# Contrato: transform(df) NUNCA muta FeatureState. Toda mutação ocorre em fit() ou update(event).

import json
import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from core.features.decay import LazyDecayCounter, LazyDecaySet


def compute_param_hash(params: Dict[str, Any]) -> str:
    """Calcula hash determinístico SHA-256 dos parâmetros aprendidos."""
    try:
        # Serialização determinística com sort_keys e tratamento padrão de tipos
        encoded = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        # Fallback para pickle se houver objetos complexos não serializáveis em JSON
        encoded = pickle.dumps(params, protocol=4)
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class FeatureState:
    """
    Estado causal do sistema de extração de features no instante anterior S_{t-}.
    
    Campos:
    - counters: Dicionário de contadores causais com decaimento exponencial.
    - seen_sets: Dicionário de conjuntos causais com TTL.
    - params: Parâmetros e baselines aprendidos offline no fit().
    - feature_version: Versão do pipeline de features.
    - param_hash: SHA-256 dos parâmetros aprendidos.
    - fitted_at: Timestamp ISO 8601 do momento do ajuste.
    """
    counters: Dict[str, LazyDecayCounter] = field(default_factory=dict)
    seen_sets: Dict[str, LazyDecaySet] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    feature_version: str = "1.0.0"
    param_hash: str = ""
    fitted_at: str = ""

    def __post_init__(self):
        if not self.fitted_at:
            self.fitted_at = datetime.now(timezone.utc).isoformat()
        if not self.param_hash and self.params:
            self.param_hash = compute_param_hash(self.params)

    def serialize(self) -> bytes:
        """
        Serializa o FeatureState em bytes para armazenamento persistente
        (compatível com pickle / trainer_meta.pkl).
        """
        payload = {
            "counters": {
                k: (v.snapshot(), v.half_life_seconds)
                for k, v in self.counters.items()
            },
            "seen_sets": {
                k: v.snapshot()
                for k, v in self.seen_sets.items()
            },
            "params": self.params,
            "feature_version": self.feature_version,
            "param_hash": self.param_hash,
            "fitted_at": self.fitted_at,
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def deserialize(cls, blob: bytes) -> "FeatureState":
        """Reconstrói uma instância de FeatureState a partir do blob de bytes."""
        payload = pickle.loads(blob)
        
        counters = {
            k: LazyDecayCounter.from_snapshot(snap, half_life)
            for k, (snap, half_life) in payload.get("counters", {}).items()
        }
        seen_sets = {
            k: LazyDecaySet.from_snapshot(data)
            for k, data in payload.get("seen_sets", {}).items()
        }
        
        return cls(
            counters=counters,
            seen_sets=seen_sets,
            params=payload.get("params", {}),
            feature_version=payload.get("feature_version", "1.0.0"),
            param_hash=payload.get("param_hash", ""),
            fitted_at=payload.get("fitted_at", ""),
        )

    def clone(self) -> "FeatureState":
        """Retorna uma cópia desacoplada do estado atual."""
        return self.deserialize(self.serialize())
