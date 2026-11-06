# core/features/decay.py
#
# Estruturas de dados causais com decaimento exponencial preguiçoso (lazy decay)
# e conjuntos com TTL (Time-To-Live).
#
# Princípios:
# - Causalidade estrita: transform() não muta o estado.
# - Atualização sob demanda: decaimento avaliado no instante da leitura t.
# - Serializável para compor o FeatureState.

import math
from typing import Tuple, Optional, Any, Dict

# Constantes de configuração recomendadas (em segundos)
SECS_PER_HOUR = 3600.0
SECS_PER_DAY = 86400.0

HALF_LIFE_RARITY_PROCESS_REGISTRY = 7.0 * SECS_PER_DAY   # 7 dias
HALF_LIFE_HOST_AFFINITY           = 7.0 * SECS_PER_DAY   # 7 dias
HALF_LIFE_AUTH_BURSTS             = 1.0 * SECS_PER_HOUR  # 1 hora
TTL_SEEN_SETS                     = 30.0 * SECS_PER_DAY  # 30 dias


class LazyDecayCounter:
    """
    Contador contínuo com decaimento exponencial sob demanda (lazy evaluation).
    
    A intensidade decai segundo e^(-gamma * (t - t_last)), onde:
    gamma = ln(2) / half_life_seconds.
    """

    def __init__(self, half_life_seconds: float):
        if half_life_seconds <= 0:
            raise ValueError(f"half_life_seconds deve ser positivo, recebido: {half_life_seconds}")
        self.half_life_seconds: float = float(half_life_seconds)
        self.gamma: float = math.log(2.0) / self.half_life_seconds
        self.value: float = 0.0
        self.t_last: Optional[float] = None

    def read(self, t: float) -> float:
        """
        Retorna o valor do contador projetado para o instante t aplicando e^(-gamma * dt).
        Não altera o estado interno (leitura pura).
        """
        if self.t_last is None:
            return 0.0
        t_val = float(t)
        if t_val <= self.t_last:
            # Não decai no passado ou instante simultâneo
            return float(self.value)
        dt = t_val - self.t_last
        return float(self.value * math.exp(-self.gamma * dt))

    def incr(self, t: float, amount: float = 1.0) -> None:
        """
        Avança o contador para o instante t com decaimento preguiçoso e adiciona amount.
        Muta o estado interno. Deve ser invocado apenas na fase de atualização (update/fit).
        """
        t_val = float(t)
        current = self.read(t_val)
        self.value = current + float(amount)
        self.t_last = t_val if self.t_last is None else max(self.t_last, t_val)

    def snapshot(self) -> Tuple[float, Optional[float]]:
        """Exporta tupla (value, t_last) para persistência."""
        return (float(self.value), float(self.t_last) if self.t_last is not None else None)

    @classmethod
    def from_snapshot(cls, snap: Tuple[float, Optional[float]], half_life_seconds: float) -> "LazyDecayCounter":
        """Reconstrói um LazyDecayCounter a partir do snapshot salvo."""
        counter = cls(half_life_seconds=half_life_seconds)
        counter.value = float(snap[0])
        counter.t_last = float(snap[1]) if snap[1] is not None else None
        return counter

    def __repr__(self) -> str:
        return f"LazyDecayCounter(value={self.value:.4f}, t_last={self.t_last}, half_life={self.half_life_seconds}s)"


class LazyDecaySet:
    """
    Conjunto com Time-To-Live (TTL) preguiçoso para rastreamento de novidade
    (ex.: is_new_user_on_host, is_new_ip_for_user).
    
    Um item expira se (t_current - t_seen) > ttl_seconds.
    """

    def __init__(self, ttl_seconds: float):
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds deve ser positivo, recebido: {ttl_seconds}")
        self.ttl_seconds: float = float(ttl_seconds)
        self._items: Dict[Any, float] = {}  # key -> last_seen_t
        self._current_t: float = 0.0

    def __contains__(self, key: Any) -> bool:
        """
        Verifica se a chave está presente e não expirou em relação a self._current_t.
        Remove chaves expiradas sob demanda.
        """
        if key not in self._items:
            return False
        last_seen = self._items[key]
        if (self._current_t - last_seen) > self.ttl_seconds:
            del self._items[key]
            return False
        return True

    def contains_at(self, key: Any, t: float) -> bool:
        """
        Avaliação estritamente causal no instante t.
        Retorna True se key foi vista em t' <= t e (t - t') <= ttl_seconds.
        Não altera o estado interno.
        """
        if key not in self._items:
            return False
        t_val = float(t)
        last_seen = self._items[key]
        if t_val < last_seen:
            # O item só foi visto no futuro em relação a t
            return False
        if (t_val - last_seen) > self.ttl_seconds:
            return False
        return True

    def add(self, key: Any, t: float) -> None:
        """
        Adiciona ou atualiza a chave com o timestamp t.
        Muta o estado interno e executa compactação preguiçosa periódica.
        """
        t_val = float(t)
        self._items[key] = t_val
        if t_val > self._current_t:
            self._current_t = t_val

        # Compactação periódica para evitar vazamento de memória em streams longos
        if len(self._items) > 5000 and len(self._items) % 1000 == 0:
            self.prune(self._current_t)

    def prune(self, current_t: float) -> int:
        """Remove explicitamente todas as chaves expiradas em current_t."""
        expired = [k for k, last_t in self._items.items() if (current_t - last_t) > self.ttl_seconds]
        for k in expired:
            del self._items[k]
        return len(expired)

    def __len__(self) -> int:
        return len(self._items)

    def snapshot(self) -> Dict[str, Any]:
        """Exporta dicionário para persistência."""
        return {
            "ttl_seconds": self.ttl_seconds,
            "items": dict(self._items),
            "current_t": self._current_t,
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "LazyDecaySet":
        """Reconstrói o LazyDecaySet a partir de um snapshot salvo."""
        decay_set = cls(ttl_seconds=data["ttl_seconds"])
        decay_set._items = dict(data.get("items", {}))
        decay_set._current_t = float(data.get("current_t", 0.0))
        return decay_set

    def __repr__(self) -> str:
        return f"LazyDecaySet(items={len(self._items)}, ttl={self.ttl_seconds}s, current_t={self._current_t})"
