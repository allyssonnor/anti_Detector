# tests/features/test_pillars.py
#
# Testes unitários para todos os novos builders da Fase 3 (Pilares 1 a 6).

import math
import numpy as np
import pandas as pd
import pytest

from core.features.builders.pilar1_identity import build_pilar1_identity
from core.features.builders.pilar2_process import build_pilar2_process, fit_pilar2_state
from core.features.builders.pilar3_auth import build_pilar3_auth, fit_pilar3_state
from core.features.builders.pilar4_topology import build_pilar4_topology, fit_pilar4_state
from core.features.builders.pilar5_temporal import build_pilar5_temporal, fit_pilar5_state
from core.features.builders.pilar6_forensic import (
    build_pilar6_forensic,
    decode_userassist,
    shannon_entropy
)


# =====================================================================
# 1. PILAR 1: IDENTIDADE E SESSÃO
# =====================================================================

def test_pilar1_entity_type_and_sessions():
    df = pd.DataFrame([
        {"UserID": "normal_user", "Time": 1000.0, "EventID": 4624, "LogonID": "0x1"},
        {"UserID": "WORKSTATION01$", "Time": 1050.0, "EventID": 4624, "LogonID": "0x2"},
        {"UserID": "SYSTEM", "Time": 1100.0, "EventID": 4672, "LogonID": "0x3"},
        {"UserID": "svc_backup", "Time": 1200.0, "EventID": 4648, "LogonID": "0x4"},
    ])
    
    out = build_pilar1_identity(df)
    
    # 4 dimensões one-hot
    assert out.loc[0, "entity_type_user"] == 1.0
    assert out.loc[1, "entity_type_machine"] == 1.0
    assert out.loc[2, "entity_type_system"] == 1.0
    assert out.loc[3, "entity_type_service"] == 1.0
    
    # Flags de privilégios e credenciais alternativas
    assert out.loc[2, "privileged_token_flag"] == 1.0
    assert out.loc[3, "alternate_credentials_flag"] == 1.0
    
    # session_age_scaled
    assert out.loc[0, "session_age_available"] == 1.0


# =====================================================================
# 2. PILAR 2: PROCESSOS
# =====================================================================

def test_pilar2_process_and_lolbins():
    train_df = pd.DataFrame([
        {"ProcessName": "explorer.exe", "ParentProcessName": "userinit.exe"},
        {"ProcessName": "chrome.exe", "ParentProcessName": "explorer.exe"},
    ] * 20)
    
    test_df = pd.DataFrame([
        {"ProcessName": "chrome.exe", "ParentProcessName": "explorer.exe"},
        {"ProcessName": "powershell.exe", "ParentProcessName": "word.exe"},  # LOLBin anômalo
    ])
    
    state = fit_pilar2_state(train_df)
    out = build_pilar2_process(test_df, state)
    
    assert out.loc[1, "process_surprisal"] > out.loc[0, "process_surprisal"]
    assert out.loc[1, "lolbin_contextual"] > 0.0
    assert out.loc[0, "lolbin_contextual"] == 0.0


# =====================================================================
# 3. PILAR 3: AUTENTICAÇÃO E FSR
# =====================================================================

def test_pilar3_fsr_and_fail_then_success():
    # Sequência: 3 falhas nos últimos 600s, seguidas de 1 sucesso em t=500
    df = pd.DataFrame([
        {"UserID": "u1", "Time": 100.0, "EventID": 4625},
        {"UserID": "u1", "Time": 200.0, "EventID": 4625},
        {"UserID": "u1", "Time": 300.0, "EventID": 4625},
        {"UserID": "u1", "Time": 500.0, "EventID": 4624},
    ])
    
    out = build_pilar3_auth(df)
    
    # No evento 4 (sucesso em t=500), prev_fails_600s == 3 -> flag deve ativar
    assert out.loc[3, "fail_then_success_flag"] == 1.0
    assert out.loc[0, "fail_then_success_flag"] == 0.0
    
    # FSR como fração real de falhas N_fail / (N_succ + N_fail + 1e-6)
    # Com 3 falhas e 0 sucessos prévios, FSR deve ser próximo de 1.0
    assert out.loc[3, "fsr_1h"] >= 0.99


# =====================================================================
# 4. PILAR 4: TOPOLOGIA E LAZY DECAY SET
# =====================================================================

def test_pilar4_topology_new_user_host():
    train_df = pd.DataFrame([
        {"UserID": "u1", "LogHost": "HOST_USUAL", "Time": 100.0},
    ] * 10)
    
    test_df = pd.DataFrame([
        {"UserID": "u1", "LogHost": "HOST_USUAL", "Time": 200.0},
        {"UserID": "u1", "LogHost": "HOST_UNSEEN", "Time": 250.0},
    ])
    
    state = fit_pilar4_state(train_df)
    out = build_pilar4_topology(test_df, state)
    
    assert out.loc[0, "is_new_user_on_host"] == 0.0
    assert out.loc[1, "is_new_user_on_host"] == 1.0
    # Escala de bits (-log2 P): host não visto tem maior surprisal (afinidade menor)
    assert out.loc[0, "host_affinity"] < out.loc[1, "host_affinity"]


# =====================================================================
# 5. PILAR 5: HARMÔNICOS E VON MISES
# =====================================================================

def test_pilar5_harmonics_and_dormancy():
    # Dois eventos: um normal, outro 24h depois (dormência > 12h)
    df = pd.DataFrame([
        {"UserID": "u1", "Time": 1700000000.0},
        {"UserID": "u1", "Time": 1700000000.0 + 86400.0},
    ])
    
    out = build_pilar5_temporal(df)
    
    # 6 dimensões de harmônicos diários
    assert "time_sin_k1" in out.columns and "time_cos_k3" in out.columns
    # 2 dimensões de harmônicos semanais
    assert "weekday_sin" in out.columns and "weekday_cos" in out.columns
    # Primeiro evento sem histórico prévio -> 0.0
    assert out.loc[0, "first_event_after_dormant"] == 0.0
    # Dormência detectada no segundo evento
    assert out.loc[1, "first_event_after_dormant"] == 1.0


# =====================================================================
# 6. PILAR 6: FORENSE, USERASSIST E REGEX EVASION
# =====================================================================

def test_pilar6_cmd_regex_and_userassist():
    # Fixture de comando malicioso obfuscado com Base64 e aspas
    cmd_obfuscated = 'powershell.exe -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAKQA= -w hidden "c""m""d"'
    df = pd.DataFrame([{"CommandLine": cmd_obfuscated}])
    
    out = build_pilar6_forensic(df)
    
    # Entropia calculada
    assert out.loc[0, "cmd_obfuscation_entropy"] > 3.0
    # Flags de evasão ativadas (Flag 1: Base64; Flag 2: Aspas; Flag 9: Janela oculta)
    assert out.loc[0, "cmd_evasion_flag_1"] == 1.0
    assert out.loc[0, "cmd_evasion_flag_2"] == 1.0
    assert out.loc[0, "cmd_evasion_flag_9"] == 1.0
