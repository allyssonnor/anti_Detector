# tests/features/test_estimators.py
#
# Testes especializados para os estimadores probabilísticos da Fase 3:
# - Von Mises circular univariado regularizado
# - Jelinek-Mercer interpolação linear com EM
# - Processo pontual autoexcitante de Hawkes

import math
import numpy as np
import pytest

from core.features.estimators.von_mises import VonMisesEstimator
from core.features.estimators.jelinek_mercer import JelinekMercerEstimator
from core.features.estimators.hawkes import HawkesEstimator


# =====================================================================
# 1. TESTES VON MISES
# =====================================================================

def test_von_mises_n1_regularization():
    """Quando N=1 ou N=0, kappa DEVE ser 0 (máxima incerteza)."""
    est = VonMisesEstimator()
    mu, kappa, n = est.fit_sample(np.array([14.0]))
    assert n == 1
    assert kappa == 0.0, "N=1 deve produzir kappa=0.0 para evitar falso pico determinístico"
    
    surp = est.surprisal(hour=14.0, mu=mu, kappa=kappa)
    assert math.isclose(surp, math.log(2.0 * math.pi), rel_tol=1e-5)


def test_von_mises_numerical_stability():
    """Verifica estabilidade em kappa extremo [1e-6, 1e3]."""
    est = VonMisesEstimator()
    for kappa in [1e-6, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        s_usual = est.surprisal(hour=12.0, mu=math.pi, kappa=kappa)
        s_unusual = est.surprisal(hour=0.0, mu=math.pi, kappa=kappa)
        assert not math.isnan(s_usual)
        assert not math.isinf(s_usual)
        assert not math.isnan(s_unusual)
        assert not math.isinf(s_unusual)
        if kappa > 1.0:
            assert s_unusual > s_usual, "Surprisal fora do pico deve ser estritamente maior que no pico"


# =====================================================================
# 2. TESTES JELINEK-MERCER
# =====================================================================

def test_jelinek_mercer_weights_sum_to_one():
    """Pesos de interpolação devem somar 1.0 e ser não-negativos."""
    jm = JelinekMercerEstimator()
    p_u = np.array([0.8, 0.7, 0.9, 0.1, 0.5])
    p_h = np.array([0.5, 0.6, 0.4, 0.2, 0.3])
    p_g = np.array([0.2, 0.2, 0.3, 0.1, 0.2])
    
    w1, w2, w3 = jm.estimate_weights_em(p_u, p_h, p_g)
    assert math.isclose(w1 + w2 + w3, 1.0, abs_tol=1e-6)
    assert w1 >= 0 and w2 >= 0 and w3 >= 0


def test_jelinek_mercer_fallback():
    """Fallback documentado (0.6, 0.3, 0.1) em caso de dados vazios."""
    jm = JelinekMercerEstimator()
    w1, w2, w3 = jm.estimate_weights_em(np.array([]), np.array([]), np.array([]))
    assert (w1, w2, w3) == (0.6, 0.3, 0.1)


# =====================================================================
# 3. TESTES HAWKES
# =====================================================================

def test_hawkes_stability_alpha_tau():
    """Garantia analítica de que alpha * tau < 1.0 (subcrítico) sempre."""
    hawkes = HawkesEstimator(tau=300.0)
    # Rajada artificial de 20 eventos em 2 minutos
    times = np.linspace(1000.0, 1120.0, 20)
    mu, alpha = hawkes.fit_user(times)
    
    branching = alpha * hawkes.tau
    assert branching < 1.0, f"Branching ratio {branching} viola a condição de subcriticidade!"
    assert mu >= hawkes.mu_min, f"Taxa basal {mu} abaixo de mu_min!"


def test_hawkes_burst_z_score():
    """Em rajada concentrada, Z-score de Hawkes deve ser nitidamente positivo."""
    hawkes = HawkesEstimator(tau=300.0)
    # Histórico de fundo espaçado
    history = [1000.0, 5000.0, 9000.0, 13000.0, 17000.0]
    mu, alpha = hawkes.fit_user(np.array(history))
    
    # Rajada súbita em t=20000 com 5 eventos em 10s
    burst = [20000.0, 20002.0, 20004.0, 20007.0, 20010.0]
    intensity = hawkes.compute_intensity(20010.0, history + burst[:-1], mu, alpha)
    z = hawkes.z_score(intensity, mu, alpha)
    
    assert z > 1.5, f"Z-score esperado em rajada deve ser > 1.5, obtido: {z}"
