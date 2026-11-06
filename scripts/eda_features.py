#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA de atributos — read-only sobre os parquets + metadados já produzidos.
NÃO treina, NÃO retreina, NÃO toca no teste (usa apenas treino para as descritivas).

Uso:
  # 1) descobrir caches disponíveis
  python scripts/eda_features.py --discover

  # 2) rodar num cenário específico
  python scripts/eda_features.py \
      --cache_dir data/GLOBAL_CACHE_DETECTOR/Synthetic_Chronological/seed_42/chronological \
      --out results_eda/synthetic_chrono_42

  # 3) rodar em lote (vários cenários em sequência)
  python scripts/eda_features.py --batch \
      --targets Synthetic_Chronological:42:chronological \
                Synthetic_Random:42:random \
                CERT:42:chronological \
                LANL:42:chronological
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if BASE not in sys.path:
    sys.path.insert(0, BASE)

# ---------------------------------------------------------------------------
# Descoberta de caches
# ---------------------------------------------------------------------------
def discover(cache_roots=None):
    if cache_roots is None:
        data_root = os.path.join(BASE, "data")
        cache_roots = [
            os.path.join("data", name)
            for name in sorted(os.listdir(data_root))
            if name.startswith("GLOBAL_CACHE")
            and os.path.isdir(os.path.join(data_root, name))
        ] if os.path.isdir(data_root) else []
    for cache_name in cache_roots:
        root = os.path.join(BASE, cache_name)
        if not os.path.exists(root):
            print(f"\n[não existe] {cache_name}")
            continue
        print(f"\n=== {cache_name} ===")
        for ds in sorted(os.listdir(root)):
            ds_path = os.path.join(root, ds)
            if not os.path.isdir(ds_path):
                continue
            for seed_dir in sorted(os.listdir(ds_path)):
                if not seed_dir.startswith("seed_"):
                    continue
                seed_path = os.path.join(ds_path, seed_dir)
                for split in sorted(os.listdir(seed_path)):
                    split_path = os.path.join(seed_path, split)
                    if not os.path.isdir(split_path):
                        continue
                    has_old = os.path.exists(os.path.join(split_path, "t_train.parquet"))
                    has_new = os.path.exists(os.path.join(split_path, "train.parquet"))
                    fmt = "OLD" if has_old else ("NEW" if has_new else "??")
                    print(f"  {ds:45s} {seed_dir:12s} {split:15s} [{fmt}]")


# ---------------------------------------------------------------------------
# Carregamento com detecção automática de formato
# ---------------------------------------------------------------------------
def _load_cache(cache_dir):
    if os.path.exists(os.path.join(cache_dir, "t_train.parquet")):
        fmt = "OLD"
        f_train = "t_train.parquet"
        f_test  = "t_test.parquet"
        f_meta  = "processor_meta.pkl"
    elif os.path.exists(os.path.join(cache_dir, "train.parquet")):
        fmt = "NEW"
        f_train = "train.parquet"
        f_test  = "test.parquet"
        f_meta  = "n1_meta.pkl"
    else:
        raise FileNotFoundError(f"Nenhum cache reconhecido em: {cache_dir}")

    train = pd.read_parquet(os.path.join(cache_dir, f_train))
    test  = pd.read_parquet(os.path.join(cache_dir, f_test)) if os.path.exists(
        os.path.join(cache_dir, f_test)) else None

    with open(os.path.join(cache_dir, f_meta), "rb") as f:
        meta = pickle.load(f)

    # Compatibilidade entre formatos
    features = meta.get("valid_features") or meta.get("feature_columns") or []

    # Fallback defensivo: extrai colunas numéricas do parquet se os metadados estiverem vazios
    if not features:
        target_col = "Is_Anomaly"
        excluded = {"UserID", "Time", target_col, "AnomalyType", "session_id", "datetime", "user", "Username"}
        features = [c for c in train.columns if c not in excluded and pd.api.types.is_numeric_dtype(train[c])]
        if not features:
            raise ValueError("Nenhuma feature numérica detectada no parquet — abortando.")
        print(f"   ⚠️  'valid_features' vazio no meta; recuperadas {len(features)} features do parquet.")

    return fmt, train, test, meta, features


# ---------------------------------------------------------------------------
# Blocos de análise
# ---------------------------------------------------------------------------
def _descriptive_by_class(train, features, out):
    rows = []
    for f in features:
        for lab, name in [(0, "normal"), (1, "anomalia")]:
            s = train.loc[train["Is_Anomaly"] == lab, f]
            if len(s) == 0:
                continue
            rows.append(dict(
                feature=f, classe=name, n=len(s),
                mean=float(s.mean()), std=float(s.std()),
                median=float(s.median()), min=float(s.min()), max=float(s.max()),
                pct_zero=float((s == 0).mean() * 100),
                pct_nan=float(s.isna().mean() * 100),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "descriptive_by_class.csv"), index=False)
    return df


def _univariate_auc_ks(train, features, out):
    y = train["Is_Anomaly"].values
    rows = []
    for f in features:
        x = train[f].values
        if np.isnan(x).all() or len(np.unique(x[~np.isnan(x)])) < 2:
            rows.append(dict(feature=f, auc=0.5, ks_stat=0.0, ks_p=1.0, note="constante/NaN"))
            continue
        try:
            auc = roc_auc_score(y, np.nan_to_num(x))
            auc = max(auc, 1 - auc)  # direção-agnóstica
        except ValueError:
            auc = 0.5
        try:
            ks, p = ks_2samp(x[y == 0], x[y == 1])
        except Exception:
            ks, p = 0.0, 1.0
        rows.append(dict(feature=f, auc=float(auc), ks_stat=float(ks), ks_p=float(p), note=""))
    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    df.to_csv(os.path.join(out, "univariate_auc.csv"), index=False)
    return df


def _correlation_and_vif(train, features, out):
    X = train[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    corr = X.corr(method="spearman")
    corr.to_csv(os.path.join(out, "correlation_spearman.csv"))

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.7})
    plt.title("Correlação de Spearman — treino (features)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "correlation_heatmap.png"), dpi=180)
    plt.close()

    # VIF (opcional — requer statsmodels)
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        Xv = X.loc[:, X.std() > 0]
        vif_rows = []
        for i, f in enumerate(Xv.columns):
            try:
                vif_rows.append(dict(feature=f, vif=float(variance_inflation_factor(Xv.values, i))))
            except Exception:
                vif_rows.append(dict(feature=f, vif=np.nan))
        pd.DataFrame(vif_rows).sort_values("vif", ascending=False).to_csv(
            os.path.join(out, "vif.csv"), index=False)
    except ImportError:
        print("    ⚠️  statsmodels ausente — VIF pulado. `pip install statsmodels` para incluir.")


def _mutual_information(train, features, out):
    from sklearn.feature_selection import mutual_info_classif
    X = train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train["Is_Anomaly"].values
    mi = mutual_info_classif(X, y, random_state=42, discrete_features=False)
    df = pd.DataFrame({"feature": features, "mutual_info": mi}).sort_values(
        "mutual_info", ascending=False)
    df.to_csv(os.path.join(out, "mutual_information.csv"), index=False)
    return df


def _per_attack_profile(train, features, out):
    if "AnomalyType" not in train.columns:
        return None
    df = train.groupby("AnomalyType")[features].mean()
    df.to_csv(os.path.join(out, "profile_by_attack.csv"))
    return df


def _per_user_auc_and_per_attack(train, features, out):
    # Identifica coluna de identificação de usuário
    user_col = None
    for cand in ["UserID", "user", "Username", "user_id"]:
        if cand in train.columns:
            user_col = cand
            break

    # (a) AUC normalizada por usuário
    if user_col is not None:
        train_norm = train.copy()
        for f in features:
            train_norm[f] = train.groupby(user_col)[f].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )

        y = train_norm["Is_Anomaly"].values
        rows_user = []
        for f in features:
            x = np.nan_to_num(train_norm[f].values)
            try:
                auc = roc_auc_score(y, x)
                auc = max(auc, 1 - auc)
            except ValueError:
                auc = 0.5
            rows_user.append(dict(feature=f, auc_per_user=auc))

        pd.DataFrame(rows_user).sort_values("auc_per_user", ascending=False).to_csv(
            os.path.join(out, "univariate_auc_per_user.csv"), index=False
        )
    else:
        print("    ⚠️  Coluna de identificador de usuário não encontrada. Pulando AUC per-user.")

    # (b) AUC por família de ataque (event-level)
    if "AnomalyType" in train.columns:
        rows_atk = []
        for atk in train["AnomalyType"].dropna().unique():
            if str(atk).lower() == "normal":
                continue
            sub = train[train["AnomalyType"].isin([atk, "normal"])]
            if len(sub) == 0 or len(sub["AnomalyType"].unique()) < 2:
                continue
            y_atk = (sub["AnomalyType"] == atk).astype(int).values
            for f in features:
                x = np.nan_to_num(sub[f].values)
                try:
                    auc = roc_auc_score(y_atk, x)
                    auc = max(auc, 1 - auc)
                except ValueError:
                    auc = 0.5
                rows_atk.append(dict(attack=atk, feature=f, auc=auc))

        if rows_atk:
            pd.DataFrame(rows_atk).to_csv(
                os.path.join(out, "univariate_auc_per_attack.csv"), index=False
            )
    else:
        print("    ⚠️  Coluna 'AnomalyType' ausente. Pulando AUC per-attack.")


def _plot_auc_comparison(out):
    global_path = os.path.join(out, "univariate_auc.csv")
    per_user_path = os.path.join(out, "univariate_auc_per_user.csv")
    if not (os.path.exists(global_path) and os.path.exists(per_user_path)):
        return

    g = pd.read_csv(global_path)[["feature", "auc"]].rename(columns={"auc": "auc_global"})
    u = pd.read_csv(per_user_path)[["feature", "auc_per_user"]]
    merged = g.merge(u, on="feature").sort_values("auc_per_user", ascending=False)
    merged["delta"] = merged["auc_per_user"] - merged["auc_global"]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(merged))))
    y = np.arange(len(merged))
    ax.barh(y - 0.2, merged["auc_global"], height=0.4, label="AUC global", color="#4C72B0")
    ax.barh(y + 0.2, merged["auc_per_user"], height=0.4, label="AUC per-user", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(merged["feature"])
    ax.axvline(0.5, ls="--", color="gray", label="acaso (0.5)")
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC (direção-agnóstica)")
    ax.set_title("Poder discriminativo: AUC global vs. AUC normalizada por usuário")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "auc_global_vs_per_user.png"), dpi=180)
    plt.close()


def _plots(top_features, train, univ, out):
    # Violino das top-10 por AUC global
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, f in zip(axes.ravel(), top_features[:10]):
        sns.violinplot(data=train, x="Is_Anomaly", y=f, ax=ax, cut=0, inner="quartile")
        ax.set_xlabel("")
        ax.set_title(f, fontsize=10)
    plt.suptitle("Top-10 atributos por AUC univariada global — distribuição por classe (treino)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "violin_top10.png"), dpi=180)
    plt.close()

    # Barras de AUC
    plt.figure(figsize=(9, max(6, 0.3 * len(univ))))
    sns.barplot(data=univ, y="feature", x="auc", orient="h", palette="viridis")
    plt.axvline(0.5, ls="--", color="gray", label="acaso (0.5)")
    plt.title("AUC univariada por atributo (poder discriminativo isolado)")
    plt.xlim(0.45, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "univariate_auc.png"), dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run_eda(cache_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n📂 cache: {cache_dir}")
    fmt, train, test, meta, features = _load_cache(cache_dir)
    print(f"   formato  : {fmt}")
    print(f"   treino   : {train.shape}")
    print(f"   teste    : {test.shape if test is not None else 'n/a'}")
    print(f"   features : {len(features)}")

    if "Is_Anomaly" not in train.columns:
        raise ValueError("Coluna 'Is_Anomaly' ausente no treino.")

    print("   → descritivas por classe...")
    _descriptive_by_class(train, features, out_dir)

    print("   → AUC univariada + KS...")
    univ = _univariate_auc_ks(train, features, out_dir)

    print("   → correlação + VIF...")
    _correlation_and_vif(train, features, out_dir)

    print("   → informação mútua...")
    _mutual_information(train, features, out_dir)

    print("   → perfil por família de ataque...")
    _per_attack_profile(train, features, out_dir)

    print("   → AUC per-user + per-attack...")
    _per_user_auc_and_per_attack(train, features, out_dir)

    print("   → gráficos...")
    top = univ[~univ["note"].eq("constante/NaN")].head(10)["feature"].tolist()
    if top:
        _plots(top, train, univ, out_dir)

    # Gráfico comparativo AUC global vs. per-user
    _plot_auc_comparison(out_dir)

    print(f"✅ EDA concluída em: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Caminho relativo a BASE, ex: data/GLOBAL_CACHE_DETECTOR/Synthetic_Chronological/seed_42/chronological")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--targets", nargs="*", default=[],
                        help="Formato: <ds>:<seed>:<split>")
    parser.add_argument("--cache_root", type=str, default=None)
    args = parser.parse_args()

    if args.discover:
        discover([args.cache_root] if args.cache_root else None)
        return

    if args.batch and args.targets:
        for t in args.targets:
            ds, seed, split = t.split(":")
            cache_root = args.cache_root or "data/GLOBAL_CACHE_DETECTOR"
            cache_rel = f"{cache_root}/{ds}/seed_{seed}/{split}"
            out = f"results_eda/{ds}__seed{seed}__{split}"
            run_eda(os.path.join(BASE, cache_rel), os.path.join(BASE, out))
        return

    if not args.cache_dir or not args.out:
        parser.error("Use --discover, ou --cache_dir + --out, ou --batch --targets ...")

    run_eda(os.path.join(BASE, args.cache_dir), os.path.join(BASE, args.out))


if __name__ == "__main__":
    main()
