# core/features/builders/pilar6_forensic.py
#
# Pilar 6: Forense Digital, Decodificação de Registro e Detecção de Evasão em Linha de Comando.

import re
import math
import codecs
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from core.features.decay import LazyDecaySet, TTL_SEEN_SETS

# Padrões regulares para as 10 flags de evasão (Spec §3.6)
EVASION_PATTERNS = [
    re.compile(r"-(?:enc|encodedcommand)\s+[A-Za-z0-9+/=]{4,}", re.IGNORECASE),          # 1. Base64
    re.compile(r"""(?:["']\w+["']){2,}"""),                                              # 2. Aspas intercaladas
    re.compile(r"\w+\^\w+"),                                                             # 3. Circumflexo (cmd)
    re.compile(r"""['"]\s*\+\s*['"]"""),                                                # 4. Concatenação de string
    re.compile(r"%[A-Za-z0-9_]+%|\$\{env:[A-Za-z0-9_]+\}", re.IGNORECASE),              # 5. Env variables
    re.compile(r"\b(?:iex|invoke-expression)\b", re.IGNORECASE),                         # 6. IEX
    re.compile(r"\b(?:downloadstring|downloadfile|webclient)\b", re.IGNORECASE),         # 7. Download cradle
    re.compile(r"-(?:executionpolicy|ep)\s+bypass", re.IGNORECASE),                      # 8. ExecPolicy bypass
    re.compile(r"-(?:w|windowstyle)\s+hidden|-nop\b", re.IGNORECASE),                    # 9. Janela oculta / NoProfile
]

SENSITIVE_REGISTRY_KEYS = [
    r"software\\microsoft\\windows\\currentversion\\run",
    r"software\\microsoft\\windows\\currentversion\\runonce",
    r"system\\currentcontrolset\\services",
    r"system\\currentcontrolset\\control\\lsa",
    r"sam\\sam\\domains\\account\\users",
]


def shannon_entropy(s: str) -> float:
    """Calcula entropia de Shannon de uma string."""
    if not s:
        return 0.0
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    total = len(s)
    ent = 0.0
    for cnt in counts.values():
        p = cnt / total
        ent -= p * math.log2(p)
    return float(ent)


def decode_userassist(val_name: str, val_bytes: bytes) -> int:
    """
    Decodifica ROT13 na chave UserAssist e extrai a contagem de execuções
    (inteiro de 4 bytes na posição 4 do array de bytes).
    """
    try:
        decoded_name = codecs.decode(val_name, "rot_13")
        if val_bytes and len(val_bytes) >= 8:
            # Bytes 4-7 contêm o contador de execuções (little-endian)
            count = int.from_bytes(val_bytes[4:8], byteorder="little")
            return count
    except Exception:
        pass
    return 0


def build_pilar6_forensic(df: pd.DataFrame, historical_state: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Constrói as features do Pilar 6:
    - userassist_decoded_exec
    - cmd_obfuscation_entropy
    - cmd_regex_evasion (10 dimensões)
    - registry_* (rarity, new, sensitive, value_entropy, len)
    - module_* (load_rarity, user_writable, first_seen, unsigned)
    """
    if df.empty:
        return df

    N = len(df)

    # 1. Linha de comando: entropia e 10 flags de evasão
    if "CommandLine" in df.columns:
        cmds = df["CommandLine"].fillna("").astype(str)
        df["cmd_obfuscation_entropy"] = [shannon_entropy(c) for c in cmds]

        # 9 flags via regex + 1 flag de comprimento > 500
        for idx, pattern in enumerate(EVASION_PATTERNS, start=1):
            df[f"cmd_evasion_flag_{idx}"] = [1.0 if pattern.search(c) else 0.0 for c in cmds]
        df["cmd_evasion_flag_10"] = (cmds.str.len() > 500).astype(float)
    else:
        df["cmd_obfuscation_entropy"] = 0.0
        for idx in range(1, 11):
            df[f"cmd_evasion_flag_{idx}"] = 0.0

    # 2. Registro (Sysmon Event 12/13/14)
    if "RegistryPath" in df.columns:
        paths = df["RegistryPath"].fillna("").astype(str).str.lower()
        is_sens = np.zeros(N, dtype=float)
        for sens in SENSITIVE_REGISTRY_KEYS:
            is_sens = np.maximum(is_sens, paths.str.contains(sens, regex=True).astype(float))
        df["is_sensitive_key"] = is_sens
        df["registry_key_rarity"] = 0.0
        df["new_registry_key_on_host"] = 0.0
        df["registry_value_entropy"] = 0.0
        df["registry_value_len"] = 0.0
        df["userassist_decoded_exec"] = 0.0
    else:
        df["is_sensitive_key"] = 0.0
        df["registry_key_rarity"] = 0.0
        df["new_registry_key_on_host"] = 0.0
        df["registry_value_entropy"] = 0.0
        df["registry_value_len"] = 0.0
        df["userassist_decoded_exec"] = 0.0

    # 3. Módulos / DLLs (Sysmon Event 7)
    if "ModulePath" in df.columns:
        mod_paths = df["ModulePath"].fillna("").astype(str).str.lower()
        writable = mod_paths.str.contains(r"\\temp\\|\\appdata\\|\\users\\public\\", regex=True)
        df["module_from_user_writable"] = writable.astype(float)
        df["module_load_rarity"] = 0.0
        df["module_first_seen_on_host"] = 0.0
        if "SignatureStatus" in df.columns:
            df["module_unsigned_or_unknown"] = (~df["SignatureStatus"].astype(str).str.lower().isin(["valid", "signed"])).astype(float)
        else:
            df["module_unsigned_or_unknown"] = 0.0
    else:
        df["module_from_user_writable"] = 0.0
        df["module_load_rarity"] = 0.0
        df["module_first_seen_on_host"] = 0.0
        df["module_unsigned_or_unknown"] = 0.0

    return df
