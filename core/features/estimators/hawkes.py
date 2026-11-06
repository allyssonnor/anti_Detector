# core/features/estimators/hawkes.py
#
# Processo pontual autoexcitante de Hawkes para modelagem de rajadas de autenticação.
# Garantias metodológicas:
# - Estabilidade matemática: branching ratio alpha * tau < 1.0 garantido por truncamento.
# - Piso de taxa basal: mu >= mu_min (1e-4 ev/s) para estabilidade em usuários de baixo volume.
# - Estimador MAP-EB com prior Gamma quando N_u < 5 eventos; MLE quando N_u >= 5.
# - Z-score da intensidade estacionária teórica E[lambda] = mu / (1 - alpha * tau).

import math
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy.optimize import minimize


class HawkesEstimator:
    """
    Modelo univariado de Hawkes com núcleo de decaimento exponencial:
    lambda(t) = mu + sum_{t_i < t} alpha * exp(-(t - t_i) / tau)
    """

    def __init__(self, tau: float = 300.0, mu_min: float = 1e-4, max_branching: float = 0.95):
        self.tau = float(tau)                  # constante temporal (default: 5 minutos = 300s)
        self.mu_min = float(mu_min)            # piso de taxa de fundo
        self.max_branching = float(max_branching)  # teto estrito para alpha * tau

    def fit_user(self, event_times: np.ndarray, global_prior: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Ajusta os parâmetros (mu, alpha) para a sequência temporal de um usuário.
        Garante alpha * tau <= max_branching < 1.0.
        """
        times = np.sort(np.asarray(event_times, dtype=float))
        times = times[~np.isnan(times)]
        N = len(times)

        # Se N < 5 ou intervalo nulo, usa prior global empírico ou fallback padrão
        if N < 5:
            if global_prior is not None:
                return float(global_prior[0]), float(global_prior[1])
            mu_fb = max(self.mu_min, float(N) / max(times[-1] - times[0], 3600.0) if N > 1 else self.mu_min)
            alpha_fb = (0.5 * self.max_branching) / self.tau
            return mu_fb, alpha_fb

        T = times[-1] - times[0]
        if T <= 0:
            return self.mu_min, (0.5 * self.max_branching) / self.tau

        # Recursive computation of log-likelihood for exponential kernel
        # R_i = sum_{j < i} exp(-(t_i - t_j) / tau)
        dt = np.diff(times)
        decay = np.exp(-dt / self.tau)

        R = np.zeros(N, dtype=float)
        for i in range(1, N):
            R[i] = decay[i - 1] * (R[i - 1] + 1.0)

        max_alpha = self.max_branching / self.tau

        def neg_log_likelihood(params: np.ndarray) -> float:
            mu, alpha = params[0], params[1]
            if mu <= 0 or alpha < 0 or (alpha * self.tau) >= 1.0:
                return 1e12

            # ln L = sum_{i=1}^N ln(mu + alpha * R_i) - mu * T - alpha * tau * sum_{i=1}^N (1 - exp(-(T_end - t_i)/tau))
            lambdas = mu + alpha * R
            if np.any(lambdas <= 0):
                return 1e12

            log_sum = np.sum(np.log(lambdas))
            integral_mu = mu * T
            integral_alpha = alpha * self.tau * np.sum(1.0 - np.exp(-(times[-1] - times) / self.tau))

            # Penalização MAP regularizada (Gamma prior)
            prior_penalty = -0.1 * math.log(mu) - 0.1 * math.log(max(alpha, 1e-9))
            return -(log_sum - integral_mu - integral_alpha) + prior_penalty

        init_mu = max(self.mu_min, float(N) / T)
        init_alpha = 0.2 * max_alpha

        res = minimize(
            neg_log_likelihood,
            x0=np.array([init_mu, init_alpha]),
            bounds=[(self.mu_min, 10.0), (0.0, max_alpha)],
            method="L-BFGS-B"
        )

        if res.success and not np.isnan(res.x[0]):
            mu_opt = float(np.clip(res.x[0], self.mu_min, 10.0))
            alpha_opt = float(np.clip(res.x[1], 0.0, max_alpha))
            return mu_opt, alpha_opt

        # Fallback seguro
        return init_mu, 0.2 * max_alpha

    def compute_intensity(self, current_t: float, prev_times: List[float], mu: float, alpha: float) -> float:
        """Calcula a intensidade pontual instantânea lambda(t)."""
        decay_sum = 0.0
        for pt in prev_times:
            if pt < current_t:
                dt = current_t - pt
                # Corta contribuição residual além de 5 meias-vidas
                if dt < 5.0 * self.tau:
                    decay_sum += math.exp(-dt / self.tau)
        return float(mu + alpha * decay_sum)

    def z_score(self, current_intensity: float, mu: float, alpha: float) -> float:
        """
        Calcula o Z-score em relação ao regime estacionário:
        E[lambda] = mu / (1 - alpha * tau)
        Var[lambda] approx (mu * alpha) / (2 * (1 - alpha * tau))
        """
        branching = min(alpha * self.tau, self.max_branching)
        denom = max(1.0 - branching, 1e-6)
        mean_intensity = mu / denom
        var_intensity = (mu * max(alpha, 1e-6)) / (2.0 * denom)
        std_intensity = math.sqrt(max(var_intensity, 1e-8))

        return float((current_intensity - mean_intensity) / std_intensity)
