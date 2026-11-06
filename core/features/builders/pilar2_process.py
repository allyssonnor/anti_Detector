# core/features/builders/pilar2_process.py
#
# Pilar 2: Processos e Linhagem.
# Correções aplicadas (Auditoria 01/11/2026):
# - Linhagem de processos com interpolação linear Jelinek-Mercer real (tripla, par, unigrama) via EM
# - lolbin_contextual estritamente causal: 1{lolbin} * process_surprisal (sem divisão por max global de batch)

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from core.features.estimators.jelinek_mercer import JelinekMercerEstimator

# Catálogo canônico versionado de Living-off-the-Land Binaries (LOLBins)
LOLBINS = {
    "certutil.exe", "bitsadmin.exe", "mshta.exe", "rundll32.exe", "regsvr32.exe",
    "cscript.exe", "wscript.exe", "wmic.exe", "powershell.exe", "pwsh.exe",
    "cmd.exe", "schtasks.exe", "at.exe", "vssadmin.exe", "nltest.exe",
    "dsquery.exe", "net.exe", "net1.exe", "psexec.exe", "esentutl.exe",
    "fodhelper.exe", "eventvwr.exe", "installutil.exe", "msbuild.exe"
}


def fit_pilar2_state(df_normal: pd.DataFrame) -> Dict[str, Any]:
    """Ajusta tabelas de frequência de transição de processos e pesos JM no treino benigno."""
    state: Dict[str, Any] = {}
    if df_normal.empty or "ProcessName" not in df_normal.columns:
        return state

    jm = JelinekMercerEstimator()
    procs = df_normal["ProcessName"].fillna("unknown").astype(str).str.lower()
    parents = (
        df_normal["ParentProcessName"].fillna("unknown").astype(str).str.lower()
        if "ParentProcessName" in df_normal.columns
        else None
    )
    grandparents = (
        df_normal["GrandparentProcessName"].fillna("unknown").astype(str).str.lower()
        if "GrandparentProcessName" in df_normal.columns
        else None
    )

    # Unigram counts
    proc_counts = procs.value_counts().to_dict()
    total_events = len(procs)
    vocab_size = max(len(proc_counts), 10)

    state["process_unigram_counts"] = proc_counts
    state["process_total_events"] = total_events
    state["process_vocab_size"] = vocab_size

    # Transição parent -> child e triplas (grandparent, parent, child)
    if parents is not None:
        if grandparents is None:
            # Reconstrói avô de forma sequencial causal
            gp_list: List[str] = []
            tracker: Dict[str, str] = {}
            for p, c in zip(parents, procs):
                gp = tracker.get(p, "unknown")
                gp_list.append(gp)
                tracker[c] = p
            gp_series = pd.Series(gp_list, index=df_normal.index)
        else:
            gp_series = grandparents

        pair_counts = df_normal.groupby([parents, procs]).size().to_dict()
        parent_totals = parents.value_counts().to_dict()

        triple_counts = df_normal.groupby([gp_series, parents, procs]).size().to_dict()
        pair_context_totals = df_normal.groupby([gp_series, parents]).size().to_dict()

        state["process_pair_counts"] = pair_counts
        state["process_parent_totals"] = parent_totals
        state["process_triple_counts"] = triple_counts
        state["process_pair_context_totals"] = pair_context_totals

        # Estimação dos pesos JM via EM
        sample_size = min(len(df_normal), 20000)
        gp_list_sample = gp_series.iloc[:sample_size].tolist()
        parents_sample = parents.iloc[:sample_size].tolist()
        procs_sample = procs.iloc[:sample_size].tolist()
        p_triples = []
        p_pairs = []
        p_unigrams = []

        for idx in range(sample_size):
            g = gp_list_sample[idx]
            p = parents_sample[idx]
            c = procs_sample[idx]

            t_cnt = triple_counts.get((g, p, c), 0)
            t_tot = pair_context_totals.get((g, p), 0)
            p_triples.append(jm.laplace_smooth_prob(t_cnt, t_tot, vocab_size))

            p_cnt = pair_counts.get((p, c), 0)
            p_tot = parent_totals.get(p, 0)
            p_pairs.append(jm.laplace_smooth_prob(p_cnt, p_tot, vocab_size))

            u_cnt = proc_counts.get(c, 0)
            p_unigrams.append(jm.laplace_smooth_prob(u_cnt, total_events, vocab_size))

        w_triple, w_pair, w_unigram = jm.estimate_weights_em(
            np.array(p_triples), np.array(p_pairs), np.array(p_unigrams)
        )
        state["process_jm_weights"] = (w_triple, w_pair, w_unigram)
    else:
        state["process_jm_weights"] = (0.0, 0.0, 1.0)

    return state


def build_pilar2_process(
    df: pd.DataFrame,
    historical_state: Dict[str, Any] = None,
    alpha: float = 0.2
) -> pd.DataFrame:
    """
    Constrói as features do Pilar 2:
    - process_surprisal (Laplace em parent->child ou unigrama)
    - lineage_surprisal (Jelinek-Mercer multiescala com triplas e pesos EM)
    - lolbin_contextual (detecção de LOLBins com peso de surprisal sem leakage intra-batch)
    """
    if df.empty or "ProcessName" not in df.columns:
        return df

    state = historical_state or {}
    jm = JelinekMercerEstimator(laplace_alpha=alpha)

    procs = df["ProcessName"].fillna("unknown").astype(str).str.lower()
    parents = (
        df["ParentProcessName"].fillna("unknown").astype(str).str.lower()
        if "ParentProcessName" in df.columns
        else None
    )
    grandparents = (
        df["GrandparentProcessName"].fillna("unknown").astype(str).str.lower()
        if "GrandparentProcessName" in df.columns
        else None
    )

    pair_counts = state.get("process_pair_counts", {})
    parent_totals = state.get("process_parent_totals", {})
    triple_counts = state.get("process_triple_counts", {})
    pair_context_totals = state.get("process_pair_context_totals", {})
    unigram_counts = state.get("process_unigram_counts", {})
    total_events = state.get("process_total_events", len(df))
    vocab_size = max(state.get("process_vocab_size", 50), 10)
    w_triple, w_pair, w_unigram = state.get("process_jm_weights", (0.5, 0.3, 0.2))

    surp_vals = np.zeros(len(df), dtype=float)
    lineage_vals = np.zeros(len(df), dtype=float)

    parent_tracker: Dict[str, str] = {}
    procs_list = procs.tolist()
    parents_list = parents.tolist() if parents is not None else None
    grandparents_list = grandparents.tolist() if grandparents is not None else None

    for i, proc in enumerate(procs_list):
        u_cnt = unigram_counts.get(proc, 0)
        p_unigram = jm.laplace_smooth_prob(u_cnt, total_events, vocab_size)

        if parents_list is not None:
            parent = parents_list[i]
            p_cnt = pair_counts.get((parent, proc), 0)
            p_tot = parent_totals.get(parent, 0)
            p_pair = jm.laplace_smooth_prob(p_cnt, p_tot, vocab_size)

            if grandparents_list is not None:
                gp = grandparents_list[i]
            else:
                gp = parent_tracker.get(parent, "unknown")

            t_cnt = triple_counts.get((gp, parent, proc), 0)
            t_tot = pair_context_totals.get((gp, parent), 0)
            p_triple = jm.laplace_smooth_prob(t_cnt, t_tot, vocab_size)

            parent_tracker[proc] = parent

            p_laplace_pair = p_pair
            p_jm = (w_triple * p_triple) + (w_pair * p_pair) + (w_unigram * p_unigram)
        else:
            p_laplace_pair = p_unigram
            p_jm = p_unigram

        surp_vals[i] = jm.surprisal(p_laplace_pair)
        lineage_vals[i] = -math.log2(max(p_jm, 1e-6))

    df["process_surprisal"] = surp_vals
    df["lineage_surprisal"] = lineage_vals

    # 2. lolbin_contextual (Correção P1: remoção do max de batch, sem leakage)
    is_lolbin = procs.isin(LOLBINS).astype(float)
    df["lolbin_contextual"] = is_lolbin * surp_vals

    return df
