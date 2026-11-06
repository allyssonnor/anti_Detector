#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap pareado sobre os scores salvos por inferir_lote.py.

Para cada (model_dir, dataset):
  - Recarrega o threshold salvo do treino em user_thresholds.json
  - Usa os scores.npy já calculados (forward pass único)
  - Bootstrap pareado: MESMO resample aplicado às 3 políticas
  - Thresholds (global@p e per_user@p) recalculados DENTRO de cada resample
  - CI95% para F1 de cada política e para os Δ pareados

Uso:
    python scripts/bootstrap_inferencia.py
    # Saída: results/12_inferencia_campeoes/bootstrap_results.csv
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = os.path.join(ROOT_PATH, "results", "12_inferencia_campeoes")

DATASETS = ["target_prev01", "target_prev05", "target_prev10", "target_prev30"]
N_ITER = 1000
SEED = 42
MIN_USER_WINDOWS = 5


# ---------------------------------------------------------------------
# Resolução de caminho (meta.json salva relativo)
# ---------------------------------------------------------------------
def resolve_model_dir(model_dir: str) -> str:
    if os.path.isabs(model_dir):
        return model_dir
    return os.path.join(ROOT_PATH, model_dir)


# ---------------------------------------------------------------------
# Carregar thresholds salvos do treino original
# ---------------------------------------------------------------------
def load_saved_thresholds(model_dir: str):
    model_dir = resolve_model_dir(model_dir)
    thr_path = os.path.join(model_dir, "user_thresholds.json")
    cfg_path = os.path.join(model_dir, "config_used.yaml")

    if not os.path.exists(thr_path):
        return None, None

    with open(thr_path) as f:
        user_thresholds = json.load(f)

    thr_mode = "global"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        thr_mode = cfg.get("training", {}).get("threshold_mode", "global")

    return user_thresholds, thr_mode


def compute_saved_pred(scores, users, user_thresholds, thr_mode):
    """Reproduz o y_pred das linhas saved_* do CSV."""
    gf = user_thresholds.get("__GLOBAL_FALLBACK__", 0.5)
    if thr_mode == "per_user":
        thr = np.array([user_thresholds.get(str(u), gf) for u in users])
    else:
        thr = np.full(len(scores), gf)
    return (scores > thr).astype(int)


# ---------------------------------------------------------------------
# Bootstrap pareado
# ---------------------------------------------------------------------
def bootstrap_paired(scores, y_true, users, saved_pred, p_honesto,
                     n_iter=N_ITER, seed=SEED, min_user_windows=MIN_USER_WINDOWS):
    rng = default_rng(seed)
    n = len(scores)

    out = {
        "f1_saved":    np.empty(n_iter),
        "f1_global":   np.empty(n_iter),
        "f1_per_user": np.empty(n_iter),
    }

    s_all = np.asarray(scores, dtype=np.float64)
    y_all = np.asarray(y_true, dtype=np.int8)
    u_all = np.asarray([str(u) for u in users])

    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        s, y, u = s_all[idx], y_all[idx], u_all[idx]

        # (a) saved — usa y_pred original reindexado
        out["f1_saved"][i] = f1_score(y, saved_pred[idx], zero_division=0)

        # (b) recalibragem global dentro do resample
        thr_g = np.percentile(s, p_honesto)
        out["f1_global"][i] = f1_score(
            y, (s > thr_g).astype(int), zero_division=0
        )

        # (c) recalibragem per_user dentro do resample
        thr_pu = np.full(n, thr_g)
        for uid in np.unique(u):
            m = (u == uid)
            if m.sum() >= min_user_windows:
                thr_pu[m] = np.percentile(s[m], p_honesto)
        out["f1_per_user"][i] = f1_score(
            y, (s > thr_pu).astype(int), zero_division=0
        )

    return out


def summarize(boot: dict) -> dict:
    res = {}
    for k, arr in boot.items():
        lo, hi = np.percentile(arr, [2.5, 97.5])
        res[f"{k}_mean"] = float(arr.mean())
        res[f"{k}_lo"]   = float(lo)
        res[f"{k}_hi"]   = float(hi)

    pairs = [
        ("d_recalib_global",     "f1_global",   "f1_saved"),
        ("d_switch_per_user",    "f1_per_user", "f1_saved"),
        ("d_per_user_vs_global", "f1_per_user", "f1_global"),
    ]
    for name, a_key, b_key in pairs:
        d = boot[a_key] - boot[b_key]
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[f"{name}_mean"] = float(d.mean())
        res[f"{name}_lo"]   = float(lo)
        res[f"{name}_hi"]   = float(hi)
        res[f"{name}_sig"]  = int(lo > 0 or hi < 0)

    return res


# ---------------------------------------------------------------------
# Parse metadados do caminho do modelo
# ---------------------------------------------------------------------
def parse_run_name(run: str):
    split = "chronological" if "Chronological" in run else \
            ("random" if "Random" in run else "?")
    preset = next(
        (t for t in ["MLP_Autoencoder", "RandomForest", "MLP_Single"] if t in run),
        run,
    )
    return preset, split


def parse_seed_from_model_dir(model_dir: str):
    parts = model_dir.split("/")
    for p in parts:
        if p.startswith("run_seed"):
            try:
                return int(p.replace("run_seed", ""))
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------
def main():
    rows = []
    skipped = []

    for dataset in DATASETS:
        ds_dir = os.path.join(BASE, dataset)
        if not os.path.isdir(ds_dir):
            print(f"⚠️  Dataset ausente: {ds_dir}")
            continue

        print(f"\n📂 {dataset}")

        for h in sorted(os.listdir(ds_dir)):
            d = os.path.join(ds_dir, h)
            if not os.path.isdir(d):
                continue

            needed = {"meta.json", "scores.npy", "y_true.npy", "users.npy"}
            if not needed.issubset(set(os.listdir(d))):
                skipped.append((d, "arquivos ausentes"))
                continue

            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)

            model_dir = meta["model_dir"]
            resolved = resolve_model_dir(model_dir)

            user_thresholds, thr_mode = load_saved_thresholds(model_dir)
            if user_thresholds is None:
                skipped.append((resolved, "user_thresholds.json ausente"))
                continue

            try:
                scores = np.load(os.path.join(d, "scores.npy")).astype(np.float64)
                y_true = np.load(os.path.join(d, "y_true.npy")).astype(np.int8)
                users  = np.load(os.path.join(d, "users.npy"), allow_pickle=True)
            except Exception as e:
                skipped.append((d, f"erro carregando .npy: {e}"))
                continue

            saved_pred = compute_saved_pred(scores, users, user_thresholds, thr_mode)

            prev = float(y_true.mean())
            p_honesto = 100.0 * (1.0 - prev)

            boot = bootstrap_paired(
                scores, y_true, users, saved_pred, p_honesto
            )
            stats = summarize(boot)

            run = os.path.basename(resolved)
            preset, split = parse_run_name(run)
            seed = parse_seed_from_model_dir(model_dir)

            row = {
                "dataset": dataset,
                "preset": preset,
                "split": split,
                "seed": seed,
                "run_name": run,
                "saved_thr_mode": thr_mode,
                "n_windows": int(len(y_true)),
                "n_anomalies": int(y_true.sum()),
                "prev_janela": round(prev, 4),
                "p_honesto": round(p_honesto, 1),
                **stats,
            }
            rows.append(row)

            print(
                f"  ✅ {preset:16s} {split:13s} seed={str(seed):4s} | "
                f"prev={prev:.3f} | "
                f"Δrecalib={row['d_recalib_global_mean']:+.4f} "
                f"[{row['d_recalib_global_lo']:+.4f},{row['d_recalib_global_hi']:+.4f}] "
                f"sig={row['d_recalib_global_sig']}"
            )

    df = pd.DataFrame(rows)
    out_path = os.path.join(BASE, "bootstrap_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n📄 Resultados: {out_path}  ({len(df)} linhas)")

    if skipped:
        print(f"\n⚠️  Pulados ({len(skipped)}):")
        for path, why in skipped:
            print(f"   - {path}: {why}")

    return df


if __name__ == "__main__":
    main()
