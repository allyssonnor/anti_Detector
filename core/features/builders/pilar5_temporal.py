# core/features/builders/pilar5_temporal.py
#
# Pilar 5: Ciclos Temporais, Harmônicos e Densidade Circular de Von Mises.
# Correções aplicadas (Auditoria 01/11/2026):
# - first_event_after_dormant: primeiro evento histórico do usuário é 0.0 (sem eventos anteriores)

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from core.features.estimators.von_mises import VonMisesEstimator


def fit_pilar5_state(df_normal: pd.DataFrame) -> Dict[str, Any]:
    """Ajusta parâmetros circulares de Von Mises por usuário."""
    state: Dict[str, Any] = {}
    if df_normal.empty or "Time" not in df_normal.columns or "UserID" not in df_normal.columns:
        return state

    dt_series = pd.to_datetime(df_normal["Time"], unit="s")
    hours = dt_series.dt.hour + (dt_series.dt.minute / 60.0)

    vm = VonMisesEstimator()
    user_vm: Dict[str, Tuple[float, float, int]] = {}

    for uid, idx in df_normal.groupby("UserID", sort=False).groups.items():
        u_hours = hours.loc[idx].to_numpy(dtype=float)
        mu, kappa, n = vm.fit_sample(u_hours)
        user_vm[str(uid)] = (mu, kappa, n)

    state["von_mises_user_params"] = user_vm
    return state


def build_pilar5_temporal(df: pd.DataFrame, historical_state: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Constrói as features do Pilar 5:
    - time_harmonics (6 dimensões: k=1,2,3)
    - weekday_harmonics (2 dimensões)
    - temporal_surprisal (Von Mises univariado por usuário)
    - first_event_after_dormant (inatividade > 12h, primeiro evento sempre 0.0)
    - dormant_surprisal
    """
    if df.empty or "Time" not in df.columns:
        return df

    state = historical_state or {}
    vm = VonMisesEstimator()
    user_vm = state.get("von_mises_user_params", {})

    dt_series = pd.to_datetime(df["Time"], unit="s")
    hour_float = dt_series.dt.hour + (dt_series.dt.minute / 60.0)
    wday = dt_series.dt.dayofweek.astype(float)

    # 1. time_harmonics (6 dimensões)
    theta_day = 2.0 * math.pi * (hour_float / 24.0)
    for k in [1, 2, 3]:
        df[f"time_sin_k{k}"] = np.sin(k * theta_day)
        df[f"time_cos_k{k}"] = np.cos(k * theta_day)

    # 2. weekday_harmonics (2 dimensões)
    theta_week = 2.0 * math.pi * (wday / 7.0)
    df["weekday_sin"] = np.sin(theta_week)
    df["weekday_cos"] = np.cos(theta_week)

    # 3. temporal_surprisal (Von Mises)
    surp_vals = np.zeros(len(df), dtype=float)
    if "UserID" in df.columns:
        uids_list = df["UserID"].astype(str).tolist()
        hours_list = hour_float.tolist()
        for i, (uid, h) in enumerate(zip(uids_list, hours_list)):
            if uid in user_vm:
                mu, kappa, _ = user_vm[uid]
                surp_vals[i] = vm.surprisal(h, mu, kappa)
            else:
                surp_vals[i] = math.log(2.0 * math.pi)

    df["temporal_surprisal"] = surp_vals

    # 4. first_event_after_dormant e dormant_surprisal
    df["first_event_after_dormant"] = 0.0
    df["dormant_surprisal"] = 0.0

    if "UserID" in df.columns:
        for _, idx in df.groupby("UserID", sort=False).groups.items():
            sub_times = df.loc[idx, "Time"].to_numpy(dtype=float)
            n_events = len(sub_times)
            is_dormant = np.zeros(n_events, dtype=float)
            dormant_surp = np.zeros(n_events, dtype=float)
            if n_events > 1:
                diffs = np.diff(sub_times)
                dormant_mask = diffs > 43200.0  # gap > 12 horas
                is_dormant[1:] = dormant_mask.astype(float)
                dormant_surp[1:] = np.log1p(np.maximum(diffs / 3600.0, 0.0)) * is_dormant[1:]
            df.loc[idx, "first_event_after_dormant"] = is_dormant
            df.loc[idx, "dormant_surprisal"] = dormant_surp

    return df
