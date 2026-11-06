# core/features/spec.py
#
# Contrato formal FeatureSpec e estruturas de metadados para o pipeline UEBA.

from dataclasses import dataclass, field
from typing import Tuple, Union, Callable, Optional, Literal, List, Set, Dict, Any


@dataclass(frozen=True)
class FeatureSpec:
    """
    Especificação formal de uma feature comportamental UEBA.
    
    Campos:
    - name: Nome canônico da feature ou coluna base.
    - pillar: Pilar conceitual da especificação (1 a 6; ou 0 para auxiliares/legado).
    - required: Tupla com nomes de colunas estritamente obrigatórias para ativação.
    - optional: Tupla com colunas adicionais para enriquecimento.
    - source: Origem dos dados ("endpoint", "security_log", "both").
    - dims: Dimensão escalar gerada (int) ou função callable para cálculo dinâmico.
    - builder: Nome do método ou módulo construtor.
    - formula_ref: Referência formal da fórmula na especificação metodológica.
    - half_life: Meia-vida em segundos para contadores decaídos (se aplicável).
    - fit_strategy: Estratégia de ajuste ("offline", "online", "none").
    - anti_leakage: Garantia temporal ("static", "causal", "offline_params").
    - degraded_mode: Comportamento quando colunas opcionais faltam ou None se ausente.
    """
    name: str
    pillar: int
    required: Tuple[str, ...]
    optional: Tuple[str, ...] = ()
    source: Literal["endpoint", "security_log", "both"] = "security_log"
    dims: Union[int, Callable] = 1
    builder: str = ""
    formula_ref: str = ""
    half_life: Optional[float] = None
    fit_strategy: Literal["offline", "online", "none"] = "offline"
    anti_leakage: Literal["static", "causal", "offline_params"] = "causal"
    degraded_mode: Optional[str] = None

    def is_active(self, available_columns: Set[str]) -> bool:
        """Uma feature só é ativada se todas as suas colunas required existirem."""
        return set(self.required).issubset(available_columns)


@dataclass
class FeatureMetadata:
    """Metadados completos do pipeline de features congelados pós-fit/transform."""
    active_features: List[str] = field(default_factory=list)
    unavailable_features: List[str] = field(default_factory=list)
    feature_set: str = "FE-Full"
    feature_version: str = "1.0.0"
    param_hash: str = ""
    fitted_at: str = ""
    specs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_features": list(self.active_features),
            "unavailable_features": list(self.unavailable_features),
            "feature_set": self.feature_set,
            "feature_version": self.feature_version,
            "param_hash": self.param_hash,
            "fitted_at": self.fitted_at,
            "specs": list(self.specs),
        }
