# tests/features/test_phase1_infra.py
#
# Testes unitários para a Fase 1: Infraestrutura Causal e Contratos.

import math
import time
import pytest
from core.features.decay import LazyDecayCounter, LazyDecaySet
from core.features.state import FeatureState, compute_param_hash
from core.features.spec import FeatureSpec, FeatureMetadata


# =====================================================================
# 1. TESTES PARA LazyDecayCounter
# =====================================================================

def test_decay_counter_half_life():
    half_life = 3600.0  # 1 hora
    counter = LazyDecayCounter(half_life_seconds=half_life)
    
    t0 = 10000.0
    counter.incr(t0, amount=10.0)
    
    # Leitura pura em t0 deve ser 10.0
    assert math.isclose(counter.read(t0), 10.0, rel_tol=1e-5)
    
    # Leitura pura não muta o estado interno
    assert math.isclose(counter.read(t0), 10.0, rel_tol=1e-5)
    
    # Em t0 + half_life deve decair exatamente para a metade (5.0)
    t1 = t0 + half_life
    assert math.isclose(counter.read(t1), 5.0, rel_tol=1e-5)
    
    # Em t0 + 2*half_life deve decair para 2.5
    t2 = t0 + (2.0 * half_life)
    assert math.isclose(counter.read(t2), 2.5, rel_tol=1e-5)
    
    # Incremento em t1 adiciona ao valor já decaído (5.0 + 3.0 = 8.0)
    counter.incr(t1, amount=3.0)
    assert math.isclose(counter.read(t1), 8.0, rel_tol=1e-5)


def test_decay_counter_snapshot_restore():
    half_life = 1800.0
    counter = LazyDecayCounter(half_life_seconds=half_life)
    t0 = 5000.0
    counter.incr(t0, amount=7.5)
    
    snap = counter.snapshot()
    restored = LazyDecayCounter.from_snapshot(snap, half_life_seconds=half_life)
    
    # Comportamento do restaurado deve ser idêntico
    t_test = t0 + 1800.0
    assert math.isclose(counter.read(t_test), restored.read(t_test), rel_tol=1e-7)


def test_decay_counter_invalid_half_life():
    with pytest.raises(ValueError):
        LazyDecayCounter(half_life_seconds=0.0)
    with pytest.raises(ValueError):
        LazyDecayCounter(half_life_seconds=-10.0)


# =====================================================================
# 2. TESTES PARA LazyDecaySet
# =====================================================================

def test_decay_set_ttl_causality():
    ttl = 86400.0  # 1 dia
    seen_set = LazyDecaySet(ttl_seconds=ttl)
    
    t0 = 100000.0
    seen_set.add("SERVER_A", t0)
    
    # Em t0 está presente
    assert seen_set.contains_at("SERVER_A", t0) is True
    
    # Não viaja para o passado (em t0 - 100 não existia)
    assert seen_set.contains_at("SERVER_A", t0 - 100.0) is False
    
    # Antes do TTL expirar está presente
    assert seen_set.contains_at("SERVER_A", t0 + (ttl - 10.0)) is True
    
    # Após o TTL está expirado
    assert seen_set.contains_at("SERVER_A", t0 + ttl + 1.0) is False
    
    # Item nunca visto
    assert seen_set.contains_at("SERVER_UNKNOWN", t0) is False


def test_decay_set_snapshot_prune():
    ttl = 300.0
    s = LazyDecaySet(ttl_seconds=ttl)
    s.add("k1", 100.0)
    s.add("k2", 200.0)
    s.add("k3", 500.0)
    
    # Snapshot e restauração
    snap = s.snapshot()
    restored = LazyDecaySet.from_snapshot(snap)
    assert len(restored) == 3
    assert restored.contains_at("k1", 150.0) is True
    
    # Prune em t=550 (k1 visto em 100 expirou pois 550 - 100 = 450 > 300; k2 visto em 200 expirou pois 550 - 200 = 350 > 300)
    expired_count = restored.prune(current_t=550.0)
    assert expired_count == 2
    assert len(restored) == 1
    assert "k3" in restored._items


# =====================================================================
# 3. TESTES PARA FeatureState E HASH DETERMINÍSTICO
# =====================================================================

def test_feature_state_hash_determinism():
    p1 = {"a": 1, "b": {"x": 10, "y": 20}, "c": [1, 2, 3]}
    p2 = {"b": {"y": 20, "x": 10}, "c": [1, 2, 3], "a": 1}  # chaves desordenadas
    
    h1 = compute_param_hash(p1)
    h2 = compute_param_hash(p2)
    assert h1 == h2, "O hash de parâmetros DEVE ser determinístico e independente da ordem das chaves!"


def test_feature_state_serialization():
    counter = LazyDecayCounter(half_life_seconds=3600.0)
    counter.incr(1000.0, 5.0)
    
    decay_set = LazyDecaySet(ttl_seconds=86400.0)
    decay_set.add("HOST_01", 1000.0)
    
    state = FeatureState(
        counters={"auth_burst": counter},
        seen_sets={"user_hosts": decay_set},
        params={"mean_hour": 14.2, "std_hour": 2.1},
        feature_version="1.2.0"
    )
    
    blob = state.serialize()
    restored = FeatureState.deserialize(blob)
    
    assert restored.feature_version == "1.2.0"
    assert restored.param_hash == state.param_hash
    assert restored.params["mean_hour"] == 14.2
    assert "auth_burst" in restored.counters
    assert math.isclose(restored.counters["auth_burst"].read(1000.0), 5.0)
    assert restored.seen_sets["user_hosts"].contains_at("HOST_01", 1000.0) is True


# =====================================================================
# 4. TESTES PARA FeatureSpec
# =====================================================================

def test_feature_spec_activation():
    spec = FeatureSpec(
        name="host_affinity",
        pillar=4,
        required=("UserID", "LogHost"),
        optional=("Time",),
        source="security_log",
        dims=1,
        builder="build_topology",
        formula_ref="P4-F1",
        anti_leakage="offline_params"
    )
    
    assert spec.is_active({"UserID", "LogHost", "Time", "EventID"}) is True
    assert spec.is_active({"UserID", "Time"}) is False  # falta LogHost
    assert spec.is_active(set()) is False
    
    # Imutabilidade (frozen)
    with pytest.raises(Exception):
        spec.pillar = 5
