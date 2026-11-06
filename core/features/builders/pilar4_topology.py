# core/features/builders/pilar4_topology.py
#
# Pilar 4: Topologia de Rede e Rastreamento de Novidade com LazyDecaySet.
# Correções aplicadas (Auditoria 01/11/2026):
# - host_affinity: calculada estritamente em escala de bits: -log2(max(p_laplace, 1e-6))
# - host_entropy_z: z-score normalizado contra baseline histórico do treino por usuário (sem leakage do batch)

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from core.features.decay import LazyDecaySet, TTL_SEEN_SETS


def fit_pilar4_state(df_normal: pd.DataFrame) -> Dict[str, Any]:
    """Ajusta afinidades de host, priors de entropia e inicializa os conjuntos de novidade LazyDecaySet."""
    state: Dict[str, Any] = {}
    if df_normal.empty:
        return state

    # 1. Host Affinity (Laplace)
    if "UserID" in df_normal.columns and "LogHost" in df_normal.columns:
        user_host_counts = df_normal.groupby(["UserID", "LogHost"]).size().to_dict()
        user_totals = df_normal.groupby("UserID").size().to_dict()
        total_unique_hosts = df_normal["LogHost"].nunique()

        state["affinity_user_host_counts"] = user_host_counts
        state["affinity_user_totals"] = user_totals
        state["affinity_vocab_size"] = max(total_unique_hosts, 5)

        # Inicializa conjuntos vistos no treino com TTL
        seen_user_host = LazyDecaySet(ttl_seconds=TTL_SEEN_SETS)
        for _, row in df_normal[["UserID", "LogHost", "Time"]].iterrows():
            seen_user_host.add(f"{row['UserID']}@{row['LogHost']}", float(row["Time"]))
        state["seen_user_host"] = seen_user_host

    if "ProcessName" in df_normal.columns and "LogHost" in df_normal.columns:
        seen_proc_host = LazyDecaySet(ttl_seconds=TTL_SEEN_SETS)
        for _, row in df_normal[["ProcessName", "LogHost", "Time"]].iterrows():
            seen_proc_host.add(f"{row['ProcessName']}@{row['LogHost']}", float(row["Time"]))
        state["seen_proc_host"] = seen_proc_host

    if "UserID" in df_normal.columns and "IpAddress" in df_normal.columns:
        seen_user_ip = LazyDecaySet(ttl_seconds=TTL_SEEN_SETS)
        for _, row in df_normal[["UserID", "IpAddress", "Time"]].iterrows():
            seen_user_ip.add(f"{row['UserID']}@{row['IpAddress']}", float(row["Time"]))
        state["seen_user_ip"] = seen_user_ip

    if "UserID" in df_normal.columns and "LogonType" in df_normal.columns:
        seen_user_logon = LazyDecaySet(ttl_seconds=TTL_SEEN_SETS)
        for _, row in df_normal[["UserID", "LogonType", "Time"]].iterrows():
            seen_user_logon.add(f"{row['UserID']}@{row['LogonType']}", float(row["Time"]))
        state["seen_user_logon"] = seen_user_logon

    # 2. Baseline de host_entropy_norm por usuário no treino benigno
    if "LogHost" in df_normal.columns and "Time" in df_normal.columns and "UserID" in df_normal.columns:
        user_entropy_stats = {}
        all_entropies = []
        for uid, idx in df_normal.groupby("UserID", sort=False).groups.items():
            sub = df_normal.loc[idx, ["Time", "LogHost"]].sort_values("Time", kind="stable")
            times = sub["Time"].to_numpy(dtype=float)
            hosts = sub["LogHost"].to_numpy()
            u_entropies = []
            for i, cur_t in enumerate(times):
                left = np.searchsorted(times, cur_t - 3600.0, side="left")
                window_hosts = hosts[left:i]
                if len(window_hosts) <= 1:
                    u_entropies.append(0.0)
                else:
                    _, counts = np.unique(window_hosts, return_counts=True)
                    if len(counts) <= 1:
                        u_entropies.append(0.0)
                    else:
                        probs = counts / np.sum(counts)
                        h_shannon = -np.sum(probs * np.log2(probs))
                        h_max = math.log2(len(counts))
                        u_entropies.append(float(h_shannon / h_max))

            if u_entropies:
                m_u = float(np.mean(u_entropies))
                s_u = float(np.std(u_entropies))
                user_entropy_stats[str(uid)] = (m_u, s_u)
                all_entropies.extend(u_entropies)

        g_mean = float(np.mean(all_entropies)) if all_entropies else 0.0
        g_std = float(np.std(all_entropies)) if all_entropies and np.std(all_entropies) > 0 else 1.0
        state["host_entropy_user_stats"] = user_entropy_stats
        state["host_entropy_global_stats"] = (g_mean, g_std)

    return state


def build_pilar4_topology(
    df: pd.DataFrame,
    historical_state: Dict[str, Any] = None,
    alpha: float = 0.2
) -> pd.DataFrame:
    """
    Constrói as features do Pilar 4:
    - host_affinity (Laplace em escala de bits -log2 P)
    - is_new_user_on_host (LazyDecaySet)
    - is_new_process_on_host (LazyDecaySet)
    - is_new_ip_for_user (LazyDecaySet)
    - is_new_logontype_for_user (LazyDecaySet)
    - host_entropy_norm + host_entropy_z (janela 1 hora contra baseline de treino)
    """
    if df.empty:
        return df

    state = historical_state or {}

    # 1. Host Affinity (Laplace) em escala de bits
    if "UserID" in df.columns and "LogHost" in df.columns:
        u_h_counts = state.get("affinity_user_host_counts", {})
        u_totals = state.get("affinity_user_totals", {})
        v_size = state.get("affinity_vocab_size", 10)

        aff_list = []
        for u, h in zip(df["UserID"], df["LogHost"]):
            cnt = u_h_counts.get((u, h), 0)
            tot = u_totals.get(u, 0)
            p_laplace = (float(cnt) + alpha) / (float(tot) + (alpha * v_size))
            aff_list.append(-math.log2(max(p_laplace, 1e-6)))
        df["host_affinity"] = aff_list

        # is_new_user_on_host via LazyDecaySet
        seen_uh = state.get("seen_user_host")
        if seen_uh is not None:
            new_uh_flags = []
            for u, h, t in zip(df["UserID"], df["LogHost"], df["Time"]):
                key = f"{u}@{h}"
                is_known = seen_uh.contains_at(key, float(t))
                new_uh_flags.append(0.0 if is_known else 1.0)
            df["is_new_user_on_host"] = new_uh_flags
        else:
            df["is_new_user_on_host"] = 0.0

    # 2. is_new_process_on_host
    if "ProcessName" in df.columns and "LogHost" in df.columns:
        seen_ph = state.get("seen_proc_host")
        if seen_ph is not None:
            df["is_new_process_on_host"] = [
                0.0 if seen_ph.contains_at(f"{p}@{h}", float(t)) else 1.0
                for p, h, t in zip(df["ProcessName"], df["LogHost"], df["Time"])
            ]

    # 3. is_new_ip_for_user
    if "UserID" in df.columns and "IpAddress" in df.columns:
        seen_uip = state.get("seen_user_ip")
        if seen_uip is not None:
            df["is_new_ip_for_user"] = [
                0.0 if seen_uip.contains_at(f"{u}@{ip}", float(t)) else 1.0
                for u, ip, t in zip(df["UserID"], df["IpAddress"], df["Time"])
            ]

    # 4. is_new_logontype_for_user
    if "UserID" in df.columns and "LogonType" in df.columns:
        seen_ulg = state.get("seen_user_logon")
        if seen_ulg is not None:
            df["is_new_logontype_for_user"] = [
                0.0 if seen_ulg.contains_at(f"{u}@{lt}", float(t)) else 1.0
                for u, lt, t in zip(df["UserID"], df["LogonType"], df["Time"])
            ]

    # 5. host_entropy_norm + host_entropy_z na janela causal [t-3600s, t) contra baseline de treino
    if "LogHost" in df.columns and "Time" in df.columns and "UserID" in df.columns:
        entropy_vals = np.zeros(len(df), dtype=float)
        entropy_z_vals = np.zeros(len(df), dtype=float)
        user_entropy_stats = state.get("host_entropy_user_stats", {})
        g_mean_h, g_std_h = state.get("host_entropy_global_stats", (0.0, 1.0))

        for _, idx in df.groupby("UserID", sort=False).groups.items():
            sub = df.loc[idx, ["Time", "LogHost"]].sort_values("Time", kind="stable")
            times = sub["Time"].to_numpy(dtype=float)
            hosts = sub["LogHost"].to_numpy()
            uid = str(df.loc[idx[0], "UserID"])
            u_mean, u_std = user_entropy_stats.get(uid, (g_mean_h, g_std_h))
            eff_std = u_std if u_std > 1e-4 else max(g_std_h, 0.1)

            u_entropies = []
            counts = {}
            total_window = 0
            left_ptr = 0
            right_ptr = 0
            n_events = len(times)

            for i in range(n_events):
                cur_t = times[i]
                target_left_t = cur_t - 3600.0

                while right_ptr < i:
                    h = hosts[right_ptr]
                    counts[h] = counts.get(h, 0) + 1
                    total_window += 1
                    right_ptr += 1

                while left_ptr < right_ptr and times[left_ptr] < target_left_t:
                    h = hosts[left_ptr]
                    counts[h] -= 1
                    if counts[h] == 0:
                        del counts[h]
                    total_window -= 1
                    left_ptr += 1

                n_distinct = len(counts)
                if n_distinct <= 1 or total_window <= 1:
                    u_entropies.append(0.0)
                else:
                    h_shannon = 0.0
                    for c in counts.values():
                        p = c / total_window
                        h_shannon -= p * math.log2(p)
                    h_max = math.log2(n_distinct)
                    u_entropies.append(float(h_shannon / h_max))

            u_ent_arr = np.array(u_entropies, dtype=float)
            entropy_vals[sub.index] = u_ent_arr
            entropy_z_vals[sub.index] = ((u_ent_arr - u_mean) / eff_std).clip(-3.0, 5.0)

        df["host_entropy_norm"] = entropy_vals
        df["host_entropy_z"] = entropy_z_vals

    return df
