# core/features/builders/pilar1_identity.py
#
# Pilar 1: Identidade, Sessão e Privilégios.

import numpy as np
import pandas as pd
from typing import Dict, Any, List

SYSTEM_ACCOUNTS = {
    "SYSTEM", "LOCAL SERVICE", "NETWORK SERVICE", "ANONYMOUS LOGON",
    "NT AUTHORITY\\SYSTEM", "NT AUTHORITY\\LOCAL SERVICE", "NT AUTHORITY\\NETWORK SERVICE"
}


def build_pilar1_identity(df: pd.DataFrame, historical_state: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Constrói as features do Pilar 1:
    - entity_type (one-hot 4d: user, machine, service, system)
    - session_age_scaled + session_age_available
    - alternate_credentials_flag (EventID 4648)
    - privileged_token_flag (EventID 4672)
    - session_mismatch
    """
    if df.empty:
        return df

    # 1. entity_type (one-hot 4 dimensões)
    users = df["UserID"].astype(str)
    is_machine = users.str.endswith("$")
    is_system = users.str.upper().isin(SYSTEM_ACCOUNTS)
    is_service = users.str.lower().str.startswith(("svc_", "service_", "svc-", "svc.")) & (~is_machine)
    is_user = (~is_machine) & (~is_system) & (~is_service)

    df["entity_type_user"] = is_user.astype(float)
    df["entity_type_machine"] = is_machine.astype(float)
    df["entity_type_service"] = is_service.astype(float)
    df["entity_type_system"] = is_system.astype(float)

    # 2. session_age_scaled + session_age_available
    df["session_age_scaled"] = 0.0
    df["session_age_available"] = 0.0

    if "EventID" in df.columns and "Time" in df.columns:
        # Rastreia t_logon de 4624 por UserID
        for _, idx in df.groupby("UserID", sort=False).groups.items():
            sub = df.loc[idx, ["Time", "EventID"]]
            times = sub["Time"].to_numpy(dtype=float)
            evts = sub["EventID"].to_numpy(dtype=np.int64)

            last_logon = np.nan
            ages = np.zeros(len(times), dtype=float)
            avail = np.zeros(len(times), dtype=float)

            for i in range(len(times)):
                if evts[i] == 4624:
                    last_logon = times[i]
                if not np.isnan(last_logon) and times[i] >= last_logon:
                    age_sec = times[i] - last_logon
                    ages[i] = min(age_sec / 28800.0, 1.0)  # normalizado para 8 horas
                    avail[i] = 1.0

            df.loc[sub.index, "session_age_scaled"] = ages
            df.loc[sub.index, "session_age_available"] = avail

    # 3. alternate_credentials_flag
    if "EventID" in df.columns:
        df["alternate_credentials_flag"] = (df["EventID"] == 4648).astype(float)
        df["privileged_token_flag"] = (df["EventID"] == 4672).astype(float)
    else:
        df["alternate_credentials_flag"] = 0.0
        df["privileged_token_flag"] = 0.0

    # 4. session_mismatch
    if "LogonID" in df.columns and "UserID" in df.columns:
        # Se um LogonID for compartilhado por múltiplos UserIDs distintos
        user_by_logon = df.groupby("LogonID")["UserID"].transform("nunique")
        df["session_mismatch"] = (user_by_logon > 1).astype(float)
    else:
        df["session_mismatch"] = 0.0

    return df
