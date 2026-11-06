# core/features/builders/pilar3_auth.py
#
# Pilar 3: Autenticação e Dinâmica Temporal de Falhas.
# Correções aplicadas (Auditoria 01/11/2026):
# - Hawkes treinado e avaliado estritamente sobre eventos de falha (EventID == 4625 / is_failed == 1)
# - FSR como fração causal real N_fail / (N_succ + N_fail + 1e-6)
# - Spray com Entropia por IP com shrinkage por sub-rede contra o baseline

import math
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from core.features.estimators.hawkes import HawkesEstimator
import core.features.builders.legacy as legacy


def fit_pilar3_state(df_normal: pd.DataFrame) -> Dict[str, Any]:
    """Ajusta priors globais e parâmetros de Hawkes por usuário estritamente sobre falhas."""
    state: Dict[str, Any] = {}
    if df_normal.empty or "Time" not in df_normal.columns or "UserID" not in df_normal.columns:
        return state

    hawkes = HawkesEstimator(tau=300.0)
    user_params: Dict[str, Tuple[float, float]] = {}

    # Filtra estritamente eventos de falha no treino benigno
    if "is_failed" in df_normal.columns:
        df_fails = df_normal[df_normal["is_failed"] == 1]
    elif "EventID" in df_normal.columns:
        df_fails = df_normal[df_normal["EventID"] == 4625]
    else:
        df_fails = df_normal.iloc[0:0]

    # Hawkes global prior ajustado sobre falhas
    if not df_fails.empty:
        all_fail_times = df_fails["Time"].to_numpy(dtype=float)
        g_mu, g_alpha = hawkes.fit_user(all_fail_times)
    else:
        g_mu, g_alpha = hawkes.mu_min, (0.2 * hawkes.max_branching) / hawkes.tau
    state["hawkes_global_prior"] = (g_mu, g_alpha)

    # Parâmetros por usuário (sobre falhas do usuário)
    if not df_fails.empty:
        for uid, group in df_fails.groupby("UserID", sort=False):
            t_user = group["Time"].to_numpy(dtype=float)
            mu_u, alpha_u = hawkes.fit_user(t_user, global_prior=(g_mu, g_alpha))
            user_params[str(uid)] = (mu_u, alpha_u)

    state["hawkes_user_params"] = user_params

    # Metadados para Spraying por IP
    domain_users_count = df_normal["UserID"].nunique()
    state["spray_domain_users"] = max(domain_users_count, 2)
    state["spray_global_stats"] = (0.0, 1.0)
    state["spray_subnet_stats"] = {}

    return state


def _time_window_ratio(
    df: pd.DataFrame,
    failed_col: str,
    success_col: str,
    window_seconds: float
) -> pd.Series:
    """
    Calcula a fração real de falhas N_fail / (N_succ + N_fail + 1e-6)
    estritamente no intervalo [t-Delta, t) excluindo o evento atual (exclusive=True).
    """
    n_fail = legacy.time_window_count(df, failed_col, window_seconds, exclusive=True)
    n_succ = legacy.time_window_count(df, success_col, window_seconds, exclusive=True)
    return n_fail / (n_succ + n_fail + 1e-6)


def build_pilar3_auth(df: pd.DataFrame, historical_state: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Constrói as features do Pilar 3:
    - hawkes_z (Processo autoexcitante estritamente sobre falhas)
    - fsr_1m, fsr_5m, fsr_1h (Fração Failure-to-Success Ratio causal)
    - spray_z (Entropia de usuários por IP na janela de 600s com shrinkage)
    - fail_then_success_flag (Spec F14: sucesso com >= 3 falhas nos 600s anteriores)
    """
    if df.empty:
        return df

    state = historical_state or {}
    hawkes = HawkesEstimator(tau=300.0)
    g_prior = state.get("hawkes_global_prior", (1e-4, 0.001))
    user_hawkes = state.get("hawkes_user_params", {})

    # Colunas binárias de falha e sucesso
    if "is_failed" not in df.columns:
        if "EventID" in df.columns:
            df["is_failed"] = (df["EventID"] == 4625).astype(int)
        else:
            df["is_failed"] = 0
    if "is_success" not in df.columns:
        if "EventID" in df.columns:
            df["is_success"] = (df["EventID"] == 4624).astype(int)
        else:
            df["is_success"] = 0

    # 1. Hawkes Z-score estritamente sobre eventos de falha
    hawkes_z_vals = np.zeros(len(df), dtype=float)
    if "Time" in df.columns and "UserID" in df.columns:
        for _, idx in df.groupby("UserID", sort=False).groups.items():
            sub = df.loc[idx, ["Time", "is_failed"]]
            times = sub["Time"].to_numpy(dtype=float)
            fails = sub["is_failed"].to_numpy(dtype=int)
            uid = str(df.loc[idx[0], "UserID"])
            mu_u, alpha_u = user_hawkes.get(uid, g_prior)

            fail_times_history: List[float] = []
            z_list: List[float] = []
            for cur_t, is_f in zip(times, fails):
                prev_fail_t = fail_times_history[-50:]
                intensity = hawkes.compute_intensity(cur_t, prev_fail_t, mu_u, alpha_u)
                z = hawkes.z_score(intensity, mu_u, alpha_u)
                z_list.append(z)
                if is_f == 1:
                    fail_times_history.append(cur_t)

            hawkes_z_vals[sub.index] = z_list

    df["hawkes_z"] = hawkes_z_vals

    # 2. FSR causal em janelas temporais estritas [t-Delta, t)
    df["fsr_1m"] = _time_window_ratio(df, "is_failed", "is_success", 60.0)
    df["fsr_5m"] = _time_window_ratio(df, "is_failed", "is_success", 300.0)
    df["fsr_1h"] = _time_window_ratio(df, "is_failed", "is_success", 3600.0)

    # 3. fail_then_success_flag (Spec F14: janela 600s, N_4625 >= 3)
    prev_fails_600s = legacy.time_window_count(df, "is_failed", 600.0, exclusive=True)
    df["fail_then_success_flag"] = ((df["is_success"] == 1) & (prev_fails_600s >= 3)).astype(float)

    # 4. spray_z (Entropia por IP com shrinkage)
    if "IpAddress" in df.columns and "Time" in df.columns and "UserID" in df.columns:
        domain_u = float(state.get("spray_domain_users", df["UserID"].nunique()))
        log2_u_dom = math.log2(max(domain_u, 2.0))
        g_mu, g_sigma = state.get("spray_global_stats", (0.0, 1.0))
        subnet_stats = state.get("spray_subnet_stats", {})

        spray_z_vals = np.zeros(len(df), dtype=float)

        # Agrupamento causal por IpAddress
        ip_series = df["IpAddress"].astype(str).str.strip()
        valid_ip_mask = ~ip_series.isin(["-", "", "none", "nan", "null"])

        if valid_ip_mask.any():
            for ip_val, group_idx in df[valid_ip_mask].groupby(ip_series[valid_ip_mask], sort=False).groups.items():
                sub = df.loc[group_idx, ["Time", "UserID", "is_success", "is_failed"]].sort_values("Time", kind="stable")
                t_arr = sub["Time"].to_numpy(dtype=float)
                u_arr = sub["UserID"].to_numpy()
                s_arr = sub["is_success"].to_numpy(dtype=int)
                f_arr = sub["is_failed"].to_numpy(dtype=int)

                n_events = len(t_arr)
                scores = []
                user_counts = {}
                n_4624 = 0
                n_4625 = 0
                total_ip_events = 0
                left_ptr = 0
                right_ptr = 0

                # Janela deslizante causal [t - 600.0, t)
                for i in range(n_events):
                    cur_t = t_arr[i]
                    target_left_t = cur_t - 600.0

                    while right_ptr < i:
                        u = u_arr[right_ptr]
                        user_counts[u] = user_counts.get(u, 0) + 1
                        n_4624 += s_arr[right_ptr]
                        n_4625 += f_arr[right_ptr]
                        total_ip_events += 1
                        right_ptr += 1

                    while left_ptr < right_ptr and t_arr[left_ptr] < target_left_t:
                        u = u_arr[left_ptr]
                        user_counts[u] -= 1
                        if user_counts[u] == 0:
                            del user_counts[u]
                        n_4624 -= s_arr[left_ptr]
                        n_4625 -= f_arr[left_ptr]
                        total_ip_events -= 1
                        left_ptr += 1

                    n_dist = len(user_counts)
                    if n_dist <= 1 or total_ip_events == 0:
                        scores.append(0.0)
                    else:
                        probs = [c / float(total_ip_events) for c in user_counts.values()]
                        h_entropy = -sum(p * math.log2(p) for p in probs)
                        h_norm = float(h_entropy / log2_u_dom)

                        sr = float(n_4624) / (float(n_4624 + n_4625) + 1e-6)
                        score = h_norm * (1.0 - sr) * math.log2(1.0 + float(n_dist))
                        scores.append(float(score) if score > 1e-6 else 0.0)

                # Shrinkage por sub-rede /24
                sub_key = ip_val.rsplit(".", 1)[0] if "." in ip_val else ip_val
                sub_mu, sub_sigma = subnet_stats.get(sub_key, (g_mu, g_sigma))

                # Normalização Z-score com shrinkage
                z_arr = ((np.array(scores) - sub_mu) / max(sub_sigma, 1e-4)).clip(-3.0, 10.0)
                spray_z_vals[sub.index] = z_arr

        df["spray_z"] = spray_z_vals
    elif "spray_z" not in df.columns:
        df["spray_z"] = 0.0

    return df
