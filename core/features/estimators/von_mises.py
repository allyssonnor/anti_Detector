# core/features/estimators/von_mises.py
#
# Estimador robusto da distribuição circular de Von Mises por usuário.
# Garantias:
# - Regularização para N pequeno: N=1 -> R_bar=0, kappa=0 (distribuição uniforme circular).
# - Uso de scipy.special.i0e para evitar overflow exponencial com kappa grande.
# - Surprisal analítica e estável para kappa in [1e-6, 1e3].

import math
import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.special import i0e


class VonMisesEstimator:
    """
    Estima a densidade circular de Von Mises:
    f(theta; mu, kappa) = exp(kappa * cos(theta - mu)) / (2 * pi * I_0(kappa))
    """

    def __init__(self, kappa_min: float = 1e-6, kappa_max: float = 1000.0):
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)

    def fit_sample(self, hours: np.ndarray) -> Tuple[float, float, int]:
        """
        Ajusta (mu, kappa, N) a partir de uma série de horários [0, 24).
        Retorna (mu_rad, kappa, N).
        """
        valid_hours = np.asarray(hours, dtype=float)
        valid_hours = valid_hours[~np.isnan(valid_hours)]
        N = len(valid_hours)
        if N <= 1:
            # Caso limite: 0 ou 1 evento -> kappa=0 (máxima incerteza / uniforme)
            mu_default = (2.0 * math.pi * valid_hours[0] / 24.0) if N == 1 else 0.0
            return float(mu_default), 0.0, N

        # Conversão de horas [0, 24) para radianos [0, 2pi)
        angles = 2.0 * math.pi * (valid_hours % 24.0) / 24.0
        C = float(np.mean(np.cos(angles)))
        S = float(np.mean(np.sin(angles)))
        R_bar = math.sqrt(C * C + S * S)
        mu = math.atan2(S, C) % (2.0 * math.pi)

        # Regularização de amostra finita: (N-1)/N evita que R_bar atinja 1.0 artificialmente
        R_reg = min(R_bar * (float(N - 1) / float(N)), 0.999)

        # Aproximação empírica de Fisher/Banerjee para MLE de kappa
        if R_reg < 0.53:
            kappa = 2.0 * R_reg + (R_reg ** 3) + (5.0 / 6.0) * (R_reg ** 5)
        elif R_reg < 0.85:
            kappa = -0.4 + 1.39 * R_reg + 0.43 / max(1.0 - R_reg, 1e-6)
        else:
            denom = max((R_reg ** 3) - 4.0 * (R_reg ** 2) + 3.0 * R_reg, 1e-6)
            kappa = 1.0 / denom

        kappa = float(np.clip(kappa, self.kappa_min, self.kappa_max))
        return float(mu), kappa, N

    def surprisal(self, hour: float, mu: float, kappa: float) -> float:
        """
        Calcula a surprisal (auto-informação negativa) -ln(f(theta)) de forma numericamente estável:
        -ln(f(theta)) = ln(2*pi) + ln(i0e(kappa)) + kappa * (1 - cos(theta - mu)).
        """
        if kappa <= self.kappa_min:
            # Distribuição uniforme no círculo de raio 2*pi
            return float(math.log(2.0 * math.pi))

        theta = 2.0 * math.pi * (float(hour) % 24.0) / 24.0
        cos_diff = math.cos(theta - mu)
        # Usa i0e(kappa) = i0(kappa) * exp(-kappa)
        log_i0e = float(np.log(i0e(kappa)))
        surp = math.log(2.0 * math.pi) + log_i0e + kappa * (1.0 - cos_diff)
        return max(0.0, surp)
