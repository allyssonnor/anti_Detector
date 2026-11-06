# tests/test_dataset_canonicalize.py
#
# Testes para a Fase 5: Aliases Canônicos e Degraded Mode no DatasetAdapter.

import pandas as pd
import pytest

from core.data.dataset_adapter import DatasetAdapter


def test_canonical_aliases_config():
    config = {
        "data": {
            "canonical_aliases": {
                "comp_dest": "LogHost",
                "usr_src": "UserID",
            }
        }
    }
    adapter = DatasetAdapter(config)
    df = pd.DataFrame([{"usr_src": "user1", "comp_dest": "PC01", "Time": 1000.0}])
    
    out = adapter._normalize(df)
    assert "UserID" in out.columns
    assert "LogHost" in out.columns
    assert out.loc[0, "UserID"] == "user1"


def test_canonical_source_ip_resolution():
    adapter = DatasetAdapter({"data": {}})
    df = pd.DataFrame([{"UserID": "u1", "SourceIP": "192.168.1.10", "Time": 1000.0}])
    
    out = adapter._normalize(df)
    assert "IpAddress" in out.columns
    assert out.loc[0, "IpAddress"] == "192.168.1.10"


def test_canonical_cert_and_lanl_mapping():
    adapter = DatasetAdapter({"data": {}})
    
    # CERT mock
    cert_df = pd.DataFrame([
        {"user": "u1", "activity": "Logon", "time": 1000.0},
        {"user": "u1", "activity": "Logoff_Failed", "time": 1010.0},
    ])
    cert_out = adapter._normalize(cert_df)
    assert "is_success" in cert_out.columns and "is_failed" in cert_out.columns
    assert cert_out.loc[0, "is_success"] == 1
    assert cert_out.loc[1, "is_failed"] == 1
    
    # LANL mock
    lanl_df = pd.DataFrame([
        {"user": "u2", "auth": "Success", "time": 2000.0},
        {"user": "u2", "auth": "Fail", "time": 2010.0},
    ])
    lanl_out = adapter._normalize(lanl_df)
    assert "is_success" in lanl_out.columns and "is_failed" in lanl_out.columns
    assert lanl_out.loc[0, "is_success"] == 1
    assert lanl_out.loc[1, "is_failed"] == 1
