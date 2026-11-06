# core/features/sets.py
#
# Definição formal dos presets de features (E0 a E9).

from typing import Set, Dict, List, Any

PRESET_E0_LEGACY = "E0-legacy"
PRESET_FE_CORE = "FE-Core"
PRESET_FE_ADVANCED = "FE-Advanced"
PRESET_FE_FULL = "FE-Full"
PRESET_FE_FULL_NO_DETERMINISTIC = "FE-Full-No-Deterministic"

# Features exatas do baseline legado E0 (38 features finais)
E0_FEATURE_NAMES: List[str] = [
    "fail_rate", "fail_count_5m", "fail_count_15m", "fail_count_60m", "fail_then_success",
    "bad_pwd_rate", "bad_pwd_count_15m", "lock_rate", "lock_count_60m",
    "host_changed", "unique_hosts", "user_host_new",
    "ip_changed", "unique_ips",
    "network_logon_rate", "network_logon_count_60m",
    "event_count", "event_z",
    "rdp_rate", "rdp_count_60m",
    "is_ntlm", "kerberos_logon_without_tgt",
    "rare_process_rate", "rare_process_count_60m",
    "admin_rate", "admin_count_60m",
    "hour_dev", "hour_z", "out_of_hours",
    "LogHost_enc", "ProcessName_enc", "IpAddress_enc", "SubStatus_enc", "AuthPackage_enc",
    "ooh_with_admin_activity", "ooh_with_rdp_event", "ooh_with_is_success", "ooh_with_bad_password",
]

PRESET_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    PRESET_E0_LEGACY: {
        "description": "Baseline histórico com 38 features legadas e encoders offline.",
        "mode": "legacy",
        "pillars": set(),
        "include_legacy": True,
    },
    PRESET_FE_CORE: {
        "description": "Núcleo comportamental dos Pilares 1 a 5 sem estimadores iterativos complexos.",
        "mode": "modular",
        "pillars": {1, 2, 3, 4, 5},
        "include_legacy": False,
        "exclude_complex_estimators": True,
    },
    PRESET_FE_ADVANCED: {
        "description": "Pilares 1 a 5 com estimadores probabilísticos avançados (Hawkes, Von Mises, JM).",
        "mode": "modular",
        "pillars": {1, 2, 3, 4, 5},
        "include_legacy": False,
        "exclude_complex_estimators": False,
    },
    PRESET_FE_FULL: {
        "description": "Especificação integral cobrindo todos os 6 pilares (Identidade, Processo, Auth, Topologia, Temporal, Forense).",
        "mode": "modular",
        "pillars": {1, 2, 3, 4, 5, 6},
        "include_legacy": False,
        "exclude_complex_estimators": False,
    },
    PRESET_FE_FULL_NO_DETERMINISTIC: {
        "description": "FE-Full estritamente causal, excluindo acumuladores determinísticos de partição.",
        "mode": "modular",
        "pillars": {1, 2, 3, 4, 5, 6},
        "include_legacy": False,
        "exclude_deterministic": True,
    },
}


def normalize_preset_name(name: str) -> str:
    """Normaliza o nome do preset para chave canônica."""
    if not name:
        return PRESET_E0_LEGACY
    s = name.strip()
    lookup = {
        "e0": PRESET_E0_LEGACY,
        "e0-legacy": PRESET_E0_LEGACY,
        "legacy": PRESET_E0_LEGACY,
        "fe-core": PRESET_FE_CORE,
        "core": PRESET_FE_CORE,
        "fe-advanced": PRESET_FE_ADVANCED,
        "advanced": PRESET_FE_ADVANCED,
        "fe-full": PRESET_FE_FULL,
        "full": PRESET_FE_FULL,
        "fe-full-no-deterministic": PRESET_FE_FULL_NO_DETERMINISTIC,
        "no-deterministic": PRESET_FE_FULL_NO_DETERMINISTIC,
    }
    return lookup.get(s.lower(), s)
