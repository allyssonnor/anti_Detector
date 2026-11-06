# core/features/builders/legacy.py
#
# Implementação integral dos builders legados do baseline E0.
# NENHUMA FÓRMULA OU NOME DE COLUNA É ALTERADO.

import numpy as np
import pandas as pd
from typing import Dict, Any, List

META_COLS: List[str] = [
    "datetime", "LogHost", "ProcessName", "UserName", "SourceIP",
    "UserID", "Time", "session_id", "AnomalyType", "EventID", "hour",
    "IpAddress", "SubStatus", "AuthenticationPackageName",
    "date", "new_session", "time_diff", "sub_session_id", "chunk_id",
    "day_of_week", "time_sin", "time_cos", "is_weekend",
    "Is_Anomaly", "_ord",
    # Intermediários de cálculo em E0 (11 helpers não numéricos no vetor final)
    "is_failed", "is_success", "bad_password", "bad_username",
    "account_locked", "rdp_event", "admin_activity",
    "network_logon", "is_kerberos_logon", "recent_4768",
    "rare_process",
]


def rolling_rate(df: pd.DataFrame, source_col: str, rate_window: int = 20) -> pd.Series:
    """
    Fração de eventos do tipo `source_col` na janela dos últimos
    `rate_window` eventos por usuário. Inclui o evento atual (estado no instante t).
    """
    if df.empty or source_col not in df.columns or "UserID" not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return df.groupby("UserID", sort=False)[source_col].transform(
        lambda v: v.rolling(rate_window, min_periods=1).mean()
    )


def time_window_count(df: pd.DataFrame, event_col: str, window_seconds: float, exclusive: bool = False) -> pd.Series:
    """
    Conta eventos com event_col == 1 nos `window_seconds` anteriores a cada
    evento, por usuário. Estritamente causal.
    `exclusive=True` exclui o próprio evento atual.
    """
    out = pd.Series(0.0, index=df.index, dtype=float)
    if df.empty or event_col not in df.columns or "Time" not in df.columns or "UserID" not in df.columns:
        return out

    for _, idx in df.groupby("UserID", sort=False).groups.items():
        sub = df.loc[idx, ["Time", event_col]].sort_values("Time", kind="stable")
        times = sub["Time"].to_numpy(dtype=float)
        vals = sub[event_col].to_numpy(dtype=float)

        cumsum = np.concatenate([[0.0], np.cumsum(vals)])
        left = np.searchsorted(times, times - window_seconds, side="left")
        sums = cumsum[np.arange(len(times)) + 1] - cumsum[left]

        if exclusive:
            sums = sums - vals

        out.loc[sub.index] = sums

    return out


def fit_legacy_state(df_normal: pd.DataFrame) -> Dict[str, Any]:
    """
    Ajusta os baselines históricos estatísticos e frequências relativas
    utilizando APENAS eventos normais (Is_Anomaly == 0).
    """
    state: Dict[str, Any] = {}
    if df_normal.empty or "Time" not in df_normal.columns or "UserID" not in df_normal.columns:
        return state

    df_temp = df_normal.copy()
    df_temp["datetime"] = pd.to_datetime(df_temp["Time"], unit="s")
    user_counts = df_temp.groupby("UserID", sort=False).size()

    # 1. Frequências relativas por usuário
    if "LogHost" in df_temp.columns:
        state["loghost_freq"] = df_temp.groupby(["UserID", "LogHost"]).size() / user_counts
        state["user_hosts_seen"] = {
            str(uid): set(hosts.dropna().astype(str))
            for uid, hosts in df_temp.groupby("UserID", sort=False)["LogHost"]
        }

    if "ProcessName" in df_temp.columns:
        state["process_freq"] = df_temp.groupby(["UserID", "ProcessName"]).size() / user_counts

    if "IpAddress" in df_temp.columns:
        state["ip_freq"] = df_temp.groupby(["UserID", "IpAddress"]).size() / user_counts

    if "SubStatus" in df_temp.columns:
        state["substatus_freq"] = df_temp.groupby(["UserID", "SubStatus"]).size() / user_counts

    if "AuthenticationPackageName" in df_temp.columns:
        state["authpkg_freq"] = df_temp.groupby(["UserID", "AuthenticationPackageName"]).size() / user_counts

    # 2. Baseline temporal por usuário
    df_temp["hour"] = df_temp["datetime"].dt.hour
    state["hour_mean"] = df_temp.groupby("UserID")["hour"].mean()
    state["hour_std"] = df_temp.groupby("UserID")["hour"].std()

    # 3. Baseline de volume diário por usuário
    df_temp["date"] = df_temp["datetime"].dt.date
    daily_counts = df_temp.groupby(["UserID", "date"]).size()
    state["daily_volume_mean"] = daily_counts.groupby("UserID").mean()
    state["daily_volume_std"] = daily_counts.groupby("UserID").std()

    return state


def apply_frequency_encoders(df: pd.DataFrame, historical_state: Dict[str, Any]) -> pd.DataFrame:
    """Aplica frequency encoders aprendidos no fit."""
    encoders = [
        ("LogHost", "loghost_freq", "LogHost_enc"),
        ("ProcessName", "process_freq", "ProcessName_enc"),
        ("IpAddress", "ip_freq", "IpAddress_enc"),
        ("SubStatus", "substatus_freq", "SubStatus_enc"),
        ("AuthenticationPackageName", "authpkg_freq", "AuthPackage_enc"),
    ]
    for col, state_key, enc_col in encoders:
        if col in df.columns and state_key in historical_state:
            freq_map = historical_state[state_key].reset_index(name=enc_col)
            df["_ord"] = range(len(df))
            df = df.merge(freq_map, on=["UserID", col], how="left")
            df = df.sort_values("_ord", kind="stable").drop(columns=["_ord"]).reset_index(drop=True)
            df[enc_col] = df[enc_col].fillna(0.0)

    if "_ord" in df.columns:
        df = df.drop_duplicates(subset=["_ord"]).reset_index(drop=True)
    return df


def build_temporal(df: pd.DataFrame, historical_state: Dict[str, Any]) -> pd.DataFrame:
    """Constrói desvios e z-score de horário."""
    df["hour"] = df["datetime"].dt.hour
    if "hour_mean" in historical_state:
        mean = df["UserID"].map(historical_state["hour_mean"])
        std = df["UserID"].map(historical_state["hour_std"]).fillna(3.0).clip(lower=0.5)

        raw = (df["hour"] - mean).abs()
        df["hour_dev"] = np.minimum(raw, 24 - raw).fillna(0)  # distância circular
        df["hour_z"] = (df["hour_dev"] / std).fillna(0)
        df["out_of_hours"] = (df["hour_z"] > 2.0).astype(int)
    else:
        df["out_of_hours"] = ((df["hour"] < 7) | (df["hour"] > 21)).astype(int)
        df["hour_dev"] = 0.0
        df["hour_z"] = 0.0
    return df


def build_auth_failure(df: pd.DataFrame, rate_window: int = 20) -> pd.DataFrame:
    """Constrói contadores causais e taxas de falha de autenticação."""
    df["is_failed"] = (df["EventID"] == 4625).astype(int)
    df["is_success"] = (df["EventID"] == 4624).astype(int)

    df["fail_rate"] = rolling_rate(df, "is_failed", rate_window)
    df["fail_count_5m"] = time_window_count(df, "is_failed", 300)
    df["fail_count_15m"] = time_window_count(df, "is_failed", 900)
    df["fail_count_60m"] = time_window_count(df, "is_failed", 3600)

    prev_fails = time_window_count(df, "is_failed", 300, exclusive=True)
    df["fail_then_success"] = ((df["is_success"] == 1) & (prev_fails > 2)).astype(int)

    if "SubStatus" in df.columns:
        sub = df["SubStatus"].astype(str).str.upper()
        df["bad_password"] = ((df["EventID"] == 4625) & sub.str.contains("0XC000006A")).astype(int)
        df["bad_username"] = ((df["EventID"] == 4625) & sub.str.contains("0XC0000064")).astype(int)
        df["account_locked"] = ((df["EventID"] == 4625) & sub.str.contains("0XC0000234")).astype(int)

        df["bad_pwd_rate"] = rolling_rate(df, "bad_password", rate_window)
        df["bad_pwd_count_15m"] = time_window_count(df, "bad_password", 900)

        df["lock_rate"] = rolling_rate(df, "account_locked", rate_window)
        df["lock_count_60m"] = time_window_count(df, "account_locked", 3600)

    return df


def build_lateral(df: pd.DataFrame, historical_state: Dict[str, Any], rate_window: int = 20) -> pd.DataFrame:
    """Constrói features de movimentação lateral e troca de host/IP."""
    if "LogHost" in df.columns:
        prev_host = df.groupby("UserID", sort=False)["LogHost"].shift()
        df["host_changed"] = (
            prev_host.notna() & (df["LogHost"].astype(str) != prev_host.astype(str))
        ).astype(int)

        def _cumnunique(x):
            seen, out = set(), []
            for v in x:
                seen.add(v)
                out.append(len(seen))
            return out

        df["unique_hosts"] = df.groupby("UserID", sort=False)["LogHost"].transform(_cumnunique)

        if "user_hosts_seen" in historical_state:
            seen_map = historical_state["user_hosts_seen"]
            df["user_host_new"] = [
                0 if str(h) in seen_map.get(str(u), set()) else 1
                for u, h in zip(df["UserID"], df["LogHost"])
            ]

    if "IpAddress" in df.columns:
        prev_ip = df.groupby("UserID", sort=False)["IpAddress"].shift()
        df["ip_changed"] = (
            prev_ip.notna() & (df["IpAddress"].astype(str) != prev_ip.astype(str))
        ).astype(int)

        def _cumnunique_ip(x):
            seen, out = set(), []
            for v in x:
                seen.add(v)
                out.append(len(seen))
            return out

        df["unique_ips"] = df.groupby("UserID", sort=False)["IpAddress"].transform(_cumnunique_ip)

    if "LogonType" in df.columns:
        df["network_logon"] = (
            ((df["EventID"] == 4624) & (df["LogonType"] == 3))
            | (df["EventID"] == 4648)
        ).astype(int)
        df["network_logon_rate"] = rolling_rate(df, "network_logon", rate_window)
        df["network_logon_count_60m"] = time_window_count(df, "network_logon", 3600)

    return df


def build_volume(df: pd.DataFrame, historical_state: Dict[str, Any]) -> pd.DataFrame:
    """Constrói features de volume diário e anomalia volumétrica."""
    if "date" in df.columns:
        df["event_count"] = df.groupby(["UserID", "date"], sort=False).cumcount() + 1
        if "daily_volume_mean" in historical_state:
            mean_map = historical_state["daily_volume_mean"]
            std_map = historical_state["daily_volume_std"]
            df["vol_mean"] = df["UserID"].map(mean_map).fillna(0)
            df["vol_std"] = df["UserID"].map(std_map).fillna(1)
            df["event_z"] = ((df["event_count"] - df["vol_mean"]) / (df["vol_std"] + 1e-6)).fillna(0)
            df.drop(columns=["vol_mean", "vol_std"], inplace=True, errors="ignore")
        else:
            df["event_z"] = 0.0
    return df


def build_rdp(df: pd.DataFrame, rate_window: int = 20) -> pd.DataFrame:
    """Constrói features de acesso RDP."""
    if "LogonType" in df.columns:
        df["rdp_event"] = ((df["EventID"] == 4624) & (df["LogonType"] == 10)).astype(int)
    else:
        df["rdp_event"] = (df["EventID"] == 4624).astype(int)

    df["rdp_rate"] = rolling_rate(df, "rdp_event", rate_window)
    df["rdp_count_60m"] = time_window_count(df, "rdp_event", 3600)
    return df


def build_golden_ticket(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói features de forja Kerberos e protocolo NTLM."""
    if "AuthenticationPackageName" in df.columns:
        pkg = df["AuthenticationPackageName"].astype(str).str.lower()
        df["is_kerberos_logon"] = (
            (df["EventID"] == 4624) & pkg.str.contains("kerberos")
        ).astype(int)
        df["is_ntlm"] = pkg.str.contains("ntlm").astype(int)

        if "EventID" in df.columns:
            df["recent_4768"] = time_window_count(
                df.assign(_evt4768=(df["EventID"] == 4768).astype(int)),
                "_evt4768", 600,
            )
            df.drop(columns=["_evt4768"], inplace=True, errors="ignore")
            df["kerberos_logon_without_tgt"] = (
                (df["is_kerberos_logon"] == 1) & (df["recent_4768"] == 0)
            ).astype(int)

    return df


def build_process(df: pd.DataFrame, rate_window: int = 20) -> pd.DataFrame:
    """Constrói features de raridade de processo."""
    if "ProcessName_enc" in df.columns:
        df["rare_process"] = (df["ProcessName_enc"] < 0.02).astype(int)
        df["rare_process_rate"] = rolling_rate(df, "rare_process", rate_window)
        df["rare_process_count_60m"] = time_window_count(df, "rare_process", 3600)
    return df


def build_privilege(df: pd.DataFrame, rate_window: int = 20) -> pd.DataFrame:
    """Constrói features de ferramentas administrativas privilegiadas."""
    if "ProcessName" in df.columns:
        proc_lower = df["ProcessName"].fillna("").astype(str).str.lower()
        admin_tools = [
            "powershell.exe", "cmd.exe", "wmic.exe", "ntdsutil.exe",
            "psexec.exe", "mmc.exe", "vssadmin.exe",
        ]
        df["admin_activity"] = proc_lower.isin(admin_tools).astype(int)
        df["admin_rate"] = rolling_rate(df, "admin_activity", rate_window)
        df["admin_count_60m"] = time_window_count(df, "admin_activity", 3600)
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói interações booleanas entre fora de horário e flags de eventos."""
    if "out_of_hours" not in df.columns:
        return df

    ooh = df["out_of_hours"].astype(int)
    for feat in ["admin_activity", "rdp_event", "is_success", "bad_password"]:
        if feat in df.columns:
            df[f"ooh_with_{feat}"] = ooh * df[feat].astype(int)

    return df
