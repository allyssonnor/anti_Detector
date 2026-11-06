# core/features/registry.py
#
# Catálogo centralizado de FeatureSpec para todos os pilares e baseline legado.

from typing import List, Dict, Set, Optional, Any
from core.features.spec import FeatureSpec
from core.features.sets import (
    PRESET_E0_LEGACY,
    PRESET_FE_CORE,
    PRESET_FE_ADVANCED,
    PRESET_FE_FULL,
    PRESET_FE_FULL_NO_DETERMINISTIC,
    E0_FEATURE_NAMES,
    normalize_preset_name,
)

FEATURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "auth_failure": {
        "required": ["Time", "UserID", "EventID"],
        "builder": "build_auth_failure",
        "attacks": ["bruteforce", "credential_stuffing"]
    },
    "lateral_movement": {
        "required": ["Time", "UserID", "LogHost"],
        "builder": "build_lateral",
        "attacks": ["lateral_movement", "pass_the_hash"]
    },
    "temporal_anomaly": {
        "required": ["Time", "UserID"],
        "builder": "build_temporal",
        "attacks": ["out_of_hours"]
    },
    "volume_anomaly": {
        "required": ["Time", "UserID"],
        "builder": "build_volume",
        "attacks": ["volume_spike"]
    },
    "rdp_anomaly": {
        "required": ["EventID", "UserID"],
        "builder": "build_rdp",
        "attacks": ["rdp_anomaly"]
    },
    "golden_ticket": {
        "required": ["EventID"],
        "builder": "build_golden_ticket",
        "attacks": ["golden_ticket"]
    },
    "process_anomaly": {
        "required": ["ProcessName", "UserID"],
        "builder": "build_process",
        "attacks": ["service_abuse"]
    },
    "privilege_anomaly": {
        "required": ["UserID", "ProcessName"],
        "builder": "build_privilege",
        "attacks": ["unusual_admin"]
    }
}

# =====================================================================
# 1. ESPECIFICAÇÕES DAS FEATURES DO BASELINE LEGADO (E0)
# =====================================================================

E0_SPECS: List[FeatureSpec] = [
    # Pilar 3 - Auth Failure
    FeatureSpec("fail_rate", 3, ("Time", "UserID", "EventID"), (), "security_log", 1, "build_auth_failure", "E0-AF1", None, "offline", "static"),
    FeatureSpec("fail_count_5m", 3, ("Time", "UserID", "EventID"), (), "security_log", 1, "build_auth_failure", "E0-AF2", None, "offline", "causal"),
    FeatureSpec("fail_count_15m", 3, ("Time", "UserID", "EventID"), (), "security_log", 1, "build_auth_failure", "E0-AF3", None, "offline", "causal"),
    FeatureSpec("fail_count_60m", 3, ("Time", "UserID", "EventID"), (), "security_log", 1, "build_auth_failure", "E0-AF4", None, "offline", "causal"),
    FeatureSpec("fail_then_success", 3, ("Time", "UserID", "EventID"), (), "security_log", 1, "build_auth_failure", "E0-AF5", None, "offline", "causal"),
    FeatureSpec("bad_pwd_rate", 3, ("Time", "UserID", "EventID", "SubStatus"), (), "security_log", 1, "build_auth_failure", "E0-AF6", None, "offline", "static"),
    FeatureSpec("bad_pwd_count_15m", 3, ("Time", "UserID", "EventID", "SubStatus"), (), "security_log", 1, "build_auth_failure", "E0-AF7", None, "offline", "causal"),
    FeatureSpec("lock_rate", 3, ("Time", "UserID", "EventID", "SubStatus"), (), "security_log", 1, "build_auth_failure", "E0-AF8", None, "offline", "static"),
    FeatureSpec("lock_count_60m", 3, ("Time", "UserID", "EventID", "SubStatus"), (), "security_log", 1, "build_auth_failure", "E0-AF9", None, "offline", "causal"),
    
    # Pilar 4 - Lateral Movement / Topologia
    FeatureSpec("host_changed", 4, ("Time", "UserID", "LogHost"), (), "security_log", 1, "build_lateral", "E0-LM1", None, "offline", "static"),
    FeatureSpec("unique_hosts", 4, ("Time", "UserID", "LogHost"), (), "security_log", 1, "build_lateral", "E0-LM2", None, "offline", "static"),
    FeatureSpec("user_host_new", 4, ("Time", "UserID", "LogHost"), (), "security_log", 1, "build_lateral", "E0-LM3", None, "offline", "offline_params"),
    FeatureSpec("ip_changed", 4, ("Time", "UserID", "IpAddress"), (), "security_log", 1, "build_lateral", "E0-LM4", None, "offline", "static"),
    FeatureSpec("unique_ips", 4, ("Time", "UserID", "IpAddress"), (), "security_log", 1, "build_lateral", "E0-LM5", None, "offline", "static"),
    FeatureSpec("network_logon_rate", 4, ("Time", "UserID", "LogHost", "EventID", "LogonType"), (), "security_log", 1, "build_lateral", "E0-LM6", None, "offline", "static"),
    FeatureSpec("network_logon_count_60m", 4, ("Time", "UserID", "LogHost", "EventID", "LogonType"), (), "security_log", 1, "build_lateral", "E0-LM7", None, "offline", "causal"),
    
    # Auxiliares de Volume
    FeatureSpec("event_count", 0, ("Time", "UserID"), (), "security_log", 1, "build_volume", "E0-VO1", None, "offline", "static"),
    FeatureSpec("event_z", 0, ("Time", "UserID"), (), "security_log", 1, "build_volume", "E0-VO2", None, "offline", "offline_params"),
    
    # RDP
    FeatureSpec("rdp_rate", 4, ("Time", "UserID", "EventID"), ("LogonType",), "security_log", 1, "build_rdp", "E0-RD1", None, "offline", "static"),
    FeatureSpec("rdp_count_60m", 4, ("Time", "UserID", "EventID"), ("LogonType",), "security_log", 1, "build_rdp", "E0-RD2", None, "offline", "causal"),
    
    # Extensões E0+ (Golden Ticket / NTLM)
    FeatureSpec("is_ntlm", 3, ("EventID", "AuthenticationPackageName"), (), "security_log", 1, "build_golden_ticket", "E0-GT1", None, "offline", "static"),
    FeatureSpec("kerberos_logon_without_tgt", 3, ("Time", "UserID", "EventID", "AuthenticationPackageName"), (), "security_log", 1, "build_golden_ticket", "E0-GT2", None, "offline", "causal"),
    
    # Processos e Privilégios
    FeatureSpec("rare_process_rate", 2, ("ProcessName", "UserID"), (), "both", 1, "build_process", "E0-PR1", None, "offline", "static"),
    FeatureSpec("rare_process_count_60m", 2, ("Time", "ProcessName", "UserID"), (), "both", 1, "build_process", "E0-PR2", None, "offline", "causal"),
    FeatureSpec("admin_rate", 1, ("ProcessName", "UserID"), (), "both", 1, "build_privilege", "E0-PV1", None, "offline", "static"),
    FeatureSpec("admin_count_60m", 1, ("Time", "ProcessName", "UserID"), (), "both", 1, "build_privilege", "E0-PV2", None, "offline", "causal"),
    
    # Temporal
    FeatureSpec("hour_dev", 5, ("Time", "UserID"), (), "security_log", 1, "build_temporal", "E0-TM1", None, "offline", "offline_params"),
    FeatureSpec("hour_z", 5, ("Time", "UserID"), (), "security_log", 1, "build_temporal", "E0-TM2", None, "offline", "offline_params"),
    FeatureSpec("out_of_hours", 5, ("Time", "UserID"), (), "security_log", 1, "build_temporal", "E0-TM3", None, "offline", "offline_params"),
    
    # Frequency Encoders Offline
    FeatureSpec("LogHost_enc", 4, ("UserID", "LogHost"), (), "security_log", 1, "apply_frequency_encoders", "E0-FE1", None, "offline", "offline_params"),
    FeatureSpec("ProcessName_enc", 2, ("UserID", "ProcessName"), (), "both", 1, "apply_frequency_encoders", "E0-FE2", None, "offline", "offline_params"),
    FeatureSpec("IpAddress_enc", 4, ("UserID", "IpAddress"), (), "security_log", 1, "apply_frequency_encoders", "E0-FE3", None, "offline", "offline_params"),
    FeatureSpec("SubStatus_enc", 3, ("UserID", "SubStatus"), (), "security_log", 1, "apply_frequency_encoders", "E0-FE4", None, "offline", "offline_params"),
    FeatureSpec("AuthPackage_enc", 3, ("UserID", "AuthenticationPackageName"), (), "security_log", 1, "apply_frequency_encoders", "E0-FE5", None, "offline", "offline_params"),
    
    # Interações
    FeatureSpec("ooh_with_admin_activity", 0, ("Time", "UserID", "ProcessName"), (), "both", 1, "add_interactions", "E0-IX1", None, "offline", "static"),
    FeatureSpec("ooh_with_rdp_event", 0, ("Time", "UserID", "EventID"), (), "security_log", 1, "add_interactions", "E0-IX2", None, "offline", "static"),
    FeatureSpec("ooh_with_is_success", 0, ("Time", "UserID", "EventID"), (), "security_log", 1, "add_interactions", "E0-IX3", None, "offline", "static"),
    FeatureSpec("ooh_with_bad_password", 0, ("Time", "UserID", "EventID", "SubStatus"), (), "security_log", 1, "add_interactions", "E0-IX4", None, "offline", "static"),
]


# =====================================================================
# 2. ESPECIFICAÇÕES DAS FEATURES DA NOVA ESPECIFICAÇÃO (PILARES 1 A 6)
# =====================================================================

SPEC_PILLAR_FEATURES: List[FeatureSpec] = [
    # PILAR 1 - Identidade, Sessão e Privilégios
    FeatureSpec("entity_type", 1, ("UserID",), (), "both", 4, "pilar1_identity", "P1-F1", None, "none", "static", degraded_mode="heuristic_dollar_system"),
    FeatureSpec("session_age_scaled", 1, ("Time", "UserID"), ("LogonID", "EventID"), "security_log", 1, "pilar1_identity", "P1-F2", None, "offline", "causal", degraded_mode=None),
    FeatureSpec("session_age_available", 1, ("Time", "UserID"), ("LogonID",), "security_log", 1, "pilar1_identity", "P1-F2b", None, "none", "static"),
    FeatureSpec("alternate_credentials_flag", 1, ("EventID",), ("ProcessName",), "security_log", 1, "pilar1_identity", "P1-F3", None, "none", "static"),
    FeatureSpec("privileged_token_flag", 1, ("EventID",), ("UserID",), "security_log", 1, "pilar1_identity", "P1-F4", None, "none", "static"),
    FeatureSpec("session_mismatch", 1, ("UserID",), ("LogonID",), "security_log", 1, "pilar1_identity", "P1-F5", None, "none", "static"),

    # PILAR 2 - Processos
    FeatureSpec("process_surprisal", 2, ("ProcessName",), ("ParentProcessName",), "endpoint", 1, "pilar2_process", "P2-F1", None, "offline", "offline_params"),
    FeatureSpec("lineage_surprisal", 2, ("ProcessName", "ParentProcessName"), (), "endpoint", 1, "pilar2_process", "P2-F2", None, "offline", "offline_params"),
    FeatureSpec("lolbin_contextual", 2, ("ProcessName",), ("CommandLine",), "endpoint", 1, "pilar2_process", "P2-F3", None, "none", "static"),

    # PILAR 3 - Autenticação
    FeatureSpec("hawkes_z", 3, ("Time", "UserID"), ("EventID",), "security_log", 1, "pilar3_auth", "P3-F1", 3600.0, "online", "causal"),
    FeatureSpec("fsr_1m", 3, ("Time", "UserID"), ("EventID",), "security_log", 1, "pilar3_auth", "P3-F2", None, "none", "causal"),
    FeatureSpec("fsr_5m", 3, ("Time", "UserID"), ("EventID",), "security_log", 1, "pilar3_auth", "P3-F3", None, "none", "causal"),
    FeatureSpec("fsr_1h", 3, ("Time", "UserID"), ("EventID",), "security_log", 1, "pilar3_auth", "P3-F4", None, "none", "causal"),
    FeatureSpec("spray_z", 3, ("Time", "UserID", "IpAddress"), (), "security_log", 1, "pilar3_auth", "P3-F5", None, "offline", "causal"),
    FeatureSpec("fail_then_success_flag", 3, ("Time", "UserID"), ("EventID",), "security_log", 1, "pilar3_auth", "P3-F6", None, "none", "causal"),

    # PILAR 4 - Topologia
    FeatureSpec("host_affinity", 4, ("UserID", "LogHost"), (), "security_log", 1, "pilar4_topology", "P4-F1", 604800.0, "offline", "offline_params"),
    FeatureSpec("is_new_user_on_host", 4, ("Time", "UserID", "LogHost"), (), "security_log", 1, "pilar4_topology", "P4-F2", 2592000.0, "online", "causal"),
    FeatureSpec("is_new_process_on_host", 4, ("Time", "ProcessName", "LogHost"), (), "both", 1, "pilar4_topology", "P4-F3", 2592000.0, "online", "causal"),
    FeatureSpec("is_new_ip_for_user", 4, ("Time", "UserID", "IpAddress"), (), "security_log", 1, "pilar4_topology", "P4-F4", 2592000.0, "online", "causal"),
    FeatureSpec("is_new_logontype_for_user", 4, ("Time", "UserID", "LogonType"), (), "security_log", 1, "pilar4_topology", "P4-F5", 2592000.0, "online", "causal"),
    FeatureSpec("host_entropy_norm", 4, ("Time", "UserID", "LogHost"), (), "security_log", 1, "pilar4_topology", "P4-F6", None, "none", "causal"),
    FeatureSpec("host_entropy_z", 4, ("Time", "UserID", "LogHost"), (), "security_log", 1, "pilar4_topology", "P4-F7", None, "offline", "causal"),

    # PILAR 5 - Temporal
    FeatureSpec("time_harmonics", 5, ("Time",), (), "security_log", 6, "pilar5_temporal", "P5-F1", None, "none", "static"),
    FeatureSpec("weekday_harmonics", 5, ("Time",), (), "security_log", 2, "pilar5_temporal", "P5-F2", None, "none", "static"),
    FeatureSpec("temporal_surprisal", 5, ("Time", "UserID"), (), "security_log", 1, "pilar5_temporal", "P5-F3", None, "offline", "offline_params"),
    FeatureSpec("first_event_after_dormant", 5, ("Time", "UserID"), (), "security_log", 1, "pilar5_temporal", "P5-F4", None, "none", "causal"),
    FeatureSpec("dormant_surprisal", 5, ("Time", "UserID"), (), "security_log", 1, "pilar5_temporal", "P5-F5", None, "offline", "causal"),

    # PILAR 6 - Forense
    FeatureSpec("userassist_decoded_exec", 6, ("RegistryPath", "RegistryValue"), (), "endpoint", 1, "pilar6_forensic", "P6-F1", None, "none", "static"),
    FeatureSpec("cmd_obfuscation_entropy", 6, ("CommandLine",), (), "endpoint", 1, "pilar6_forensic", "P6-F2", None, "none", "static"),
    FeatureSpec("cmd_regex_evasion", 6, ("CommandLine",), (), "endpoint", 10, "pilar6_forensic", "P6-F3", None, "none", "static"),
    FeatureSpec("registry_key_rarity", 6, ("RegistryPath",), (), "endpoint", 1, "pilar6_forensic", "P6-F4", 604800.0, "offline", "offline_params"),
    FeatureSpec("new_registry_key_on_host", 6, ("Time", "RegistryPath", "LogHost"), (), "endpoint", 1, "pilar6_forensic", "P6-F5", 2592000.0, "online", "causal"),
    FeatureSpec("is_sensitive_key", 6, ("RegistryPath",), (), "endpoint", 1, "pilar6_forensic", "P6-F6", None, "none", "static"),
    FeatureSpec("registry_value_entropy", 6, ("RegistryValue",), (), "endpoint", 1, "pilar6_forensic", "P6-F7", None, "none", "static"),
    FeatureSpec("registry_value_len", 6, ("RegistryValue",), (), "endpoint", 1, "pilar6_forensic", "P6-F8", None, "none", "static"),
    FeatureSpec("module_load_rarity", 6, ("ModulePath",), (), "endpoint", 1, "pilar6_forensic", "P6-F9", 604800.0, "offline", "offline_params"),
    FeatureSpec("module_from_user_writable", 6, ("ModulePath",), (), "endpoint", 1, "pilar6_forensic", "P6-F10", None, "none", "static"),
    FeatureSpec("module_first_seen_on_host", 6, ("Time", "ModulePath", "LogHost"), (), "endpoint", 1, "pilar6_forensic", "P6-F11", 2592000.0, "online", "causal"),
    FeatureSpec("module_unsigned_or_unknown", 6, ("ModulePath", "SignatureStatus"), (), "endpoint", 1, "pilar6_forensic", "P6-F12", None, "none", "static", degraded_mode=None),
]


def get_specs_for_preset(preset_name: str) -> List[FeatureSpec]:
    """Retorna a lista de FeatureSpec associadas ao preset solicitado."""
    canon = normalize_preset_name(preset_name)
    if canon == PRESET_E0_LEGACY:
        return list(E0_SPECS)
    elif canon == PRESET_FE_CORE:
        # Pilares 1 a 5 sem estimadores complexos
        complex_names = {"hawkes_z", "temporal_surprisal", "lineage_surprisal"}
        return [s for s in SPEC_PILLAR_FEATURES if s.pillar in {1, 2, 3, 4, 5} and s.name not in complex_names]
    elif canon == PRESET_FE_ADVANCED:
        # Pilares 1 a 5 completos
        return [s for s in SPEC_PILLAR_FEATURES if s.pillar in {1, 2, 3, 4, 5}]
    elif canon == PRESET_FE_FULL:
        # Pilares 1 a 6 completos
        return list(SPEC_PILLAR_FEATURES)
    elif canon == PRESET_FE_FULL_NO_DETERMINISTIC:
        # Pilares 1 a 6 sem acumuladores estáticos
        return [s for s in SPEC_PILLAR_FEATURES if s.anti_leakage != "static" or s.formula_ref not in {"E0-LM2", "E0-LM5", "E0-VO1"}]
    else:
        # Default de segurança: E0-legacy
        return list(E0_SPECS)
