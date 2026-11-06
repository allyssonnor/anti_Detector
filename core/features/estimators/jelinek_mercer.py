# core/features/estimators/jelinek_mercer.py
#
# Estimador de interpolação linear Jelinek-Mercer com algoritmo EM
# e suavização de Laplace para raridade de processos, linhagens e registro.

import math
from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class JelinekMercerEstimator:
    """
    Combina modelos de probabilidade em múltiplos níveis (usuário, host, global):
    P_JM = lambda_1 * P_user + lambda_2 * P_host + lambda_3 * P_global
    onde sum(lambda_i) == 1 e lambda_i >= 0.
    """

    DEFAULT_WEIGHTS = (0.6, 0.3, 0.1)

    def __init__(self, laplace_alpha: float = 0.2, max_iter: int = 50, tol: float = 1e-4):
        self.laplace_alpha = float(laplace_alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def estimate_weights_em(
        self,
        p_user: np.ndarray,
        p_host: np.ndarray,
        p_global: np.ndarray,
        initial_weights: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Ajusta os pesos lambda_1, lambda_2, lambda_3 via Expectation-Maximization (EM).
        Se a convergência falhar ou os vetores forem vazios, utiliza o fallback documentado (0.6, 0.3, 0.1).
        """
        N = len(p_user)
        if N == 0 or len(p_host) != N or len(p_global) != N:
            return self.DEFAULT_WEIGHTS

        # Piso para evitar divisão por zero
        p1 = np.maximum(np.asarray(p_user, dtype=float), 1e-9)
        p2 = np.maximum(np.asarray(p_host, dtype=float), 1e-9)
        p3 = np.maximum(np.asarray(p_global, dtype=float), 1e-9)

        w = list(initial_weights) if initial_weights else [0.6, 0.3, 0.1]
        w = np.array(w, dtype=float)
        w /= np.sum(w)

        for _ in range(self.max_iter):
            # E-step: calcula as responsabilidades posteriores
            denom = w[0] * p1 + w[1] * p2 + w[2] * p3
            denom = np.maximum(denom, 1e-12)

            gamma1 = (w[0] * p1) / denom
            gamma2 = (w[1] * p2) / denom
            gamma3 = (w[2] * p3) / denom

            # M-step: atualiza os pesos
            new_w = np.array([np.mean(gamma1), np.mean(gamma2), np.mean(gamma3)], dtype=float)
            new_w /= np.sum(new_w)

            if np.max(np.abs(new_w - w)) < self.tol:
                break
            w = new_w

        # Garante normalização estrita sum(lambda_i) == 1
        w_final = tuple(float(x) for x in (w / np.sum(w)))
        return w_final

    def laplace_smooth_prob(self, count: int, total: int, vocab_size: int) -> float:
        """Suavização aditiva de Laplace com parâmetro alpha."""
        alpha = self.laplace_alpha
        v = max(1, vocab_size)
        return (float(count) + alpha) / (float(total) + (alpha * v))

    def surprisal(self, prob: float) -> float:
        """Calcula surprisal -log2(P)."""
        p_safe = max(prob, 1e-12)
        return float(-math.log2(p_safe))
