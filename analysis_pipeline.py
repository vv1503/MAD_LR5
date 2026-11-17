"""Пайплайн лабораторной: дескриптивный анализ, CLOPE, Gower+KMedoids, сравнение."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from clope import CLOPE
from gower import gower_matrix
from kmodes.kmodes import KModes
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "winequality-red.csv"
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "DejaVu Sans"

FEATURE_RU = {
    "fixed acidity": "Фикс. кислотность",
    "volatile acidity": "Летучая кислотность",
    "citric acid": "Лимонная кислота",
    "residual sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free sulfur dioxide": "Своб. SO2",
    "total sulfur dioxide": "Общий SO2",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь",
    "quality": "Качество",
}


def load_data() -> pd.DataFrame:
    """Загружает датасет Wine Quality (red)."""
    return pd.read_csv(DATA, sep=";")


def descriptive_analysis(df: pd.DataFrame) -> dict:
    """Строит дескриптивную статистику и сохраняет графики."""
    desc = df.describe().T
    desc["skew"] = df.skew(numeric_only=True)
    desc["kurtosis"] = df.kurtosis(numeric_only=True)
    desc.to_csv(RES / "descriptive_stats.csv", encoding="utf-8-sig")

    # Распределения
    cols = list(df.columns)
    n = len(cols)
    nrows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 3.2 * nrows))
    axes = axes.ravel()
    for i, col in enumerate(cols):
        axes[i].hist(df[col], bins=20, color="#2c7fb8", edgecolor="white", alpha=0.9)
        axes[i].set_title(FEATURE_RU.get(col, col))
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Гистограммы признаков Wine Quality", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "01_histograms.png")
    plt.close(fig)

    # Корреляции
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Корреляционная матрица признаков")
    fig.tight_layout()
    fig.savefig(FIG / "02_correlation.png")
    plt.close(fig)

    # Качество
    fig, ax = plt.subplots(figsize=(7, 4))
    vc = df["quality"].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color="#41b6c4", edgecolor="white")
    ax.set_xlabel("Оценка качества")
    ax.set_ylabel("Число наблюдений")
    ax.set_title("Распределение целевой переменной quality")
    for x, y in zip(vc.index.astype(str), vc.values):
        ax.text(x, y + 8, str(y), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "03_quality_distribution.png")
    plt.close(fig)

    # Boxplots
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.ravel()
    for i, col in enumerate(cols):
        sns.boxplot(y=df[col], ax=axes[i], color="#7fcdbb")
        axes[i].set_title(FEATURE_RU.get(col, col))
        axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Диаграммы размаха (boxplot)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "04_boxplots.png")
    plt.close(fig)

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "quality_counts": {str(k): int(v) for k, v in vc.items()},
        "desc": desc,
    }


def discretize_features(df: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """Дискретизация числовых признаков на категории Low/Medium/High."""
    labels = ["Low", "Medium", "High"][:n_bins]
    cat = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == "quality":
            cat[col] = df[col].astype(str)
            continue
        try:
            cat[col] = pd.qcut(df[col], q=n_bins, labels=labels, duplicates="drop")
        except ValueError:
            cat[col] = pd.cut(df[col], bins=n_bins, labels=labels, include_lowest=True)
        cat[col] = cat[col].astype(str)
    return cat


def to_transactions(cat_df: pd.DataFrame) -> list[list[str]]:
    """Преобразует категориальную таблицу в список транзакций CLOPE."""
    transactions = []
    for _, row in cat_df.iterrows():
        items = [f"{col}={row[col]}" for col in cat_df.columns]
        transactions.append(items)
    return transactions


def encode_categorical(cat_df: pd.DataFrame) -> np.ndarray:
    """Кодирует категории целыми числами для k-modes и метрик."""
    enc = cat_df.copy()
    for col in enc.columns:
        enc[col] = LabelEncoder().fit_transform(enc[col].astype(str))
    return enc.values


def hamming_pairwise(X: np.ndarray) -> np.ndarray:
    """Матрица расстояний Хэмминга (доля несовпадающих признаков)."""
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        diffs = (X[i] != X).mean(axis=1)
        D[i] = diffs
    return D


def cluster_feature_profiles(df: pd.DataFrame, labels: np.ndarray, name: str) -> pd.DataFrame:
    """Средние значения исходных признаков по кластерам."""
    tmp = df.copy()
    tmp["cluster"] = labels
    profile = tmp.groupby("cluster").mean(numeric_only=True)
    profile["size"] = tmp.groupby("cluster").size()
    profile.to_csv(RES / f"profile_{name}.csv", encoding="utf-8-sig")
    return profile


def feature_significance(df: pd.DataFrame, labels: np.ndarray, name: str) -> pd.DataFrame:
    """ANOVA F-статистика как оценка значимости признаков для кластеров."""
    from sklearn.feature_selection import f_classif

    X = df.drop(columns=["quality"], errors="ignore")
    # если quality в df — оставляем все числовые
    X = df.select_dtypes(include=[np.number]).copy()
    if "cluster" in X.columns:
        X = X.drop(columns=["cluster"])
    F, p = f_classif(X, labels)
    out = pd.DataFrame({"feature": X.columns, "F": F, "p_value": p})
    out = out.sort_values("F", ascending=False)
    out.to_csv(RES / f"feature_significance_{name}.csv", encoding="utf-8-sig", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(out["feature"].map(lambda c: FEATURE_RU.get(c, c)), out["F"], color="#225ea8")
    ax.invert_yaxis()
    ax.set_xlabel("F-статистика (ANOVA)")
    ax.set_title(f"Значимость признаков ({name})")
    fig.tight_layout()
    fig.savefig(FIG / f"significance_{name}.png")
    plt.close(fig)
    return out


def external_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Внешние метрики качества относительно quality."""
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "Homogeneity": float(homogeneity_score(y_true, y_pred)),
        "Completeness": float(completeness_score(y_true, y_pred)),
        "V-measure": float(v_measure_score(y_true, y_pred)),
    }


def plot_cluster_quality(df: pd.DataFrame, labels: np.ndarray, name: str) -> None:
    """Распределение quality внутри кластеров."""
    tmp = pd.DataFrame({"cluster": labels, "quality": df["quality"].values})
    ct = pd.crosstab(tmp["cluster"], tmp["quality"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_xlabel("Кластер")
    ax.set_ylabel("Доля, %")
    ax.set_title(f"Состав кластеров по quality ({name})")
    ax.legend(title="quality", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"quality_by_cluster_{name}.png")
    plt.close(fig)


def plot_profiles(profile: pd.DataFrame, name: str) -> None:
    """Тепловая карта нормализованных профилей кластеров."""
    cols = [c for c in profile.columns if c != "size"]
    Z = profile[cols].copy()
    Z = (Z - Z.mean()) / Z.std(ddof=0)
    fig, ax = plt.subplots(figsize=(11, max(3, 0.55 * len(Z))))
    sns.heatmap(Z, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"Стандартизированные профили кластеров ({name})")
    fig.tight_layout()
    fig.savefig(FIG / f"profiles_{name}.png")
    plt.close(fig)


def run_clope(transactions, y_true, df, cat_df) -> dict:
    """Подбор repulsion и финальная кластеризация CLOPE."""
    grid = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]
    rows = []
    best = None
    best_score = -1

    X_enc = encode_categorical(cat_df)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(transactions), size=min(800, len(transactions)), replace=False)
    D_sub = hamming_pairwise(X_enc[idx])

    for r in grid:
        model = CLOPE(repulsion=r, max_iter=15)
        labels = np.array(model.fit_predict(transactions))
        k = len(np.unique(labels))
        ext = external_metrics(y_true, labels)
        sil = float("nan")
        sizes = pd.Series(labels).value_counts()
        min_share = float(sizes.min() / len(labels)) if k > 0 else 0.0
        if 1 < k < len(idx):
            try:
                sil = float(silhouette_score(D_sub, labels[idx], metric="precomputed"))
            except Exception:
                sil = float("nan")
        row = {
            "r": r,
            "n_clusters": k,
            "profit": model.profit_history_[-1],
            "silhouette_hamming": sil,
            "min_cluster_share": min_share,
            **ext,
        }
        rows.append(row)
        # предпочтение интерпретируемых решений: 2..6 кластеров и невырожденные размеры
        score = ext["NMI"] + 0.15 * (0 if np.isnan(sil) else sil)
        if 2 <= k <= 6 and min_share >= 0.05 and (best is None or score > best_score):
            best_score = score
            best = (r, labels, model)

    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(RES / "clope_param_grid.csv", encoding="utf-8-sig", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(grid_df["r"], grid_df["n_clusters"], "o-", color="#2c7fb8", label="Число кластеров")
    ax1.set_xlabel("Коэффициент отталкивания r")
    ax1.set_ylabel("Число кластеров", color="#2c7fb8")
    ax2 = ax1.twinx()
    ax2.plot(grid_df["r"], grid_df["NMI"], "s--", color="#d95f0e", label="NMI")
    ax2.set_ylabel("NMI (vs quality)", color="#d95f0e")
    ax1.set_title("Подбор параметра repulsion (CLOPE)")
    fig.tight_layout()
    fig.savefig(FIG / "05_clope_param_selection.png")
    plt.close(fig)

    if best is None:
        # fallback: r=2.5
        model = CLOPE(repulsion=2.5, max_iter=15)
        labels = np.array(model.fit_predict(transactions))
        best = (2.5, labels, model)

    r_best, labels, model = best
    profile = cluster_feature_profiles(df, labels, "clope")
    plot_profiles(profile, "clope")
    plot_cluster_quality(df, labels, "clope")
    sig = feature_significance(df, labels, "clope")

    # profit history
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(model.profit_history_, "o-", color="#238b45")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Profit(C, r)")
    ax.set_title(f"Динамика Profit при r={r_best}")
    fig.tight_layout()
    fig.savefig(FIG / "06_clope_profit.png")
    plt.close(fig)

    # sizes
    sizes = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sizes.index.astype(str), sizes.values, color="#41b6c4")
    ax.set_xlabel("Кластер")
    ax.set_ylabel("Размер")
    ax.set_title(f"Размеры кластеров CLOPE (r={r_best})")
    fig.tight_layout()
    fig.savefig(FIG / "07_clope_sizes.png")
    plt.close(fig)

    metrics = {
        "method": "CLOPE",
        "params": {"r": r_best, "n_clusters": int(len(np.unique(labels)))},
        "external": external_metrics(y_true, labels),
        "profit": float(model.profit_history_[-1]),
        "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        "top_features": sig.head(5).to_dict(orient="records"),
    }
    np.save(RES / "labels_clope.npy", labels)
    return metrics, labels, grid_df


def pam_gower(Gw: np.ndarray, k: int, n_init: int = 8, max_iter: int = 25, seed: int = 42) -> np.ndarray:
    """Простой PAM (k-medoids) по готовой матрице расстояний Гауэра."""
    rng = np.random.default_rng(seed)
    n = Gw.shape[0]
    best_labels, best_cost = None, np.inf

    for _ in range(n_init):
        medoids = rng.choice(n, size=k, replace=False)
        labels = np.argmin(Gw[:, medoids], axis=1)
        for _it in range(max_iter):
            new_medoids = []
            for j in range(k):
                members = np.where(labels == j)[0]
                if len(members) == 0:
                    new_medoids.append(rng.integers(0, n))
                    continue
                # медоид — точка с минимальной суммой расстояний внутри кластера
                costs = Gw[np.ix_(members, members)].sum(axis=1)
                new_medoids.append(members[int(np.argmin(costs))])
            new_medoids = np.array(new_medoids)
            new_labels = np.argmin(Gw[:, new_medoids], axis=1)
            if np.array_equal(new_medoids, medoids) and np.array_equal(new_labels, labels):
                break
            medoids, labels = new_medoids, new_labels
        cost = float(Gw[np.arange(n), medoids[labels]].sum())
        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()
    return best_labels


def run_gower_clustering(df: pd.DataFrame, cat_df: pd.DataFrame, y_true: np.ndarray) -> dict:
    """Кластеризация на мере Гауэра (PAM) + дополнительно k-modes."""
    # quality исключён — используется только как внешняя метка
    features = df.drop(columns=["quality"])
    # все признаки числовые → Gower эквивалентен нормализованному Manhattan
    Gw = gower_matrix(features.to_numpy())
    np.save(RES / "gower_matrix.npy", Gw)

    rows = []
    best_k, best_labels, best_score = None, None, -1
    for k in range(2, 7):
        labels = pam_gower(Gw, k=k, n_init=10, max_iter=30, seed=42)
        sil = float(silhouette_score(Gw, labels, metric="precomputed"))
        ext = external_metrics(y_true, labels)
        sizes = pd.Series(labels).value_counts()
        min_share = float(sizes.min() / len(labels))
        rows.append({
            "k": k,
            "silhouette_gower": sil,
            "min_cluster_share": min_share,
            **ext,
        })
        score = sil + 0.2 * ext["NMI"]
        if min_share >= 0.05 and score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # также сохраним иерархический complete linkage для отчёта
    condensed = squareform(Gw, checks=False)
    Z = linkage(condensed, method="complete")
    hac_labels = fcluster(Z, t=best_k, criterion="maxclust") - 1
    pd.DataFrame({
        "PAM": best_labels,
        "HAC_complete": hac_labels,
    }).to_csv(RES / "gower_labels_variants.csv", index=False)

    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(RES / "gower_param_grid.csv", encoding="utf-8-sig", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(grid_df["k"], grid_df["silhouette_gower"], "o-", color="#2c7fb8", label="Silhouette")
    ax1.set_xlabel("Число кластеров k")
    ax1.set_ylabel("Silhouette (Gower)", color="#2c7fb8")
    ax2 = ax1.twinx()
    ax2.plot(grid_df["k"], grid_df["NMI"], "s--", color="#d95f0e", label="NMI")
    ax2.set_ylabel("NMI (vs quality)", color="#d95f0e")
    ax1.set_title("Подбор k для PAM по мере Гауэра")
    fig.tight_layout()
    fig.savefig(FIG / "08_gower_param_selection.png")
    plt.close(fig)

    labels = best_labels
    profile = cluster_feature_profiles(df, labels, "gower")
    plot_profiles(profile, "gower")
    plot_cluster_quality(df, labels, "gower")
    sig = feature_significance(df, labels, "gower")

    sizes = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sizes.index.astype(str), sizes.values, color="#fdae61")
    ax.set_xlabel("Кластер")
    ax.set_ylabel("Размер")
    ax.set_title(f"Размеры кластеров (Gower + PAM, k={best_k})")
    fig.tight_layout()
    fig.savefig(FIG / "09_gower_sizes.png")
    plt.close(fig)

    # k-modes на категориальных данных: подбор k
    X_cat = encode_categorical(cat_df)
    km_rows = []
    best_km, best_km_score = None, -1
    for k in range(2, 7):
        km = KModes(n_clusters=k, init="Huang", n_init=8, verbose=0, random_state=42)
        km_labels = km.fit_predict(X_cat)
        ext = external_metrics(y_true, km_labels)
        km_rows.append({"k": k, **ext})
        if ext["NMI"] > best_km_score:
            best_km_score = ext["NMI"]
            best_km = (k, km_labels)
    pd.DataFrame(km_rows).to_csv(RES / "kmodes_param_grid.csv", encoding="utf-8-sig", index=False)
    km_k, km_labels = best_km
    km_ext = external_metrics(y_true, km_labels)
    cluster_feature_profiles(df, km_labels, "kmodes")
    plot_cluster_quality(df, km_labels, "kmodes")
    feature_significance(df, km_labels, "kmodes")
    np.save(RES / "labels_kmodes.npy", km_labels)

    metrics = {
        "method": "Gower + PAM (k-medoids)",
        "params": {
            "k": int(best_k),
            "distance": "Gower",
            "silhouette": float(silhouette_score(Gw, labels, metric="precomputed")),
        },
        "external": external_metrics(y_true, labels),
        "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        "top_features": sig.head(5).to_dict(orient="records"),
        "kmodes_external": km_ext,
        "kmodes_k": int(km_k),
    }
    np.save(RES / "labels_gower.npy", labels)
    return metrics, labels, grid_df


def compare_methods(y_true, labels_clope, labels_gower, metrics_clope, metrics_gower) -> pd.DataFrame:
    """Сравнительная таблица и графики."""
    rows = []
    for name, m in [("CLOPE", metrics_clope), ("Gower+PAM", metrics_gower)]:
        row = {"method": name, **m["external"], "n_clusters": m["params"].get("n_clusters", m["params"].get("k"))}
        rows.append(row)
    # kmodes
    rows.append({
        "method": "K-modes",
        **metrics_gower["kmodes_external"],
        "n_clusters": metrics_gower["kmodes_k"],
    })
    cmp = pd.DataFrame(rows)
    cmp.to_csv(RES / "comparison_metrics.csv", encoding="utf-8-sig", index=False)

    # ARI между методами
    agree = pd.DataFrame({
        "pair": ["CLOPE vs Gower", "CLOPE vs quality", "Gower vs quality"],
        "ARI": [
            adjusted_rand_score(labels_clope, labels_gower),
            adjusted_rand_score(labels_clope, y_true),
            adjusted_rand_score(labels_gower, y_true),
        ],
        "NMI": [
            normalized_mutual_info_score(labels_clope, labels_gower),
            normalized_mutual_info_score(labels_clope, y_true),
            normalized_mutual_info_score(labels_gower, y_true),
        ],
    })
    agree.to_csv(RES / "methods_agreement.csv", encoding="utf-8-sig", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cmp))
    w = 0.25
    ax.bar(x - w, cmp["ARI"], width=w, label="ARI", color="#2c7fb8")
    ax.bar(x, cmp["NMI"], width=w, label="NMI", color="#41b6c4")
    ax.bar(x + w, cmp["V-measure"], width=w, label="V-measure", color="#7fcdbb")
    ax.set_xticks(x)
    ax.set_xticklabels(cmp["method"])
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение методов кластеризации (внешние метрики vs quality)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "10_comparison.png")
    plt.close(fig)

    return cmp, agree


def main():
    """Запуск полного пайплайна лабораторной работы."""
    print("Loading data...")
    df = load_data()
    desc_info = descriptive_analysis(df)
    print(f"N={desc_info['n_rows']}, missing={desc_info['missing']}")

    # для CLOPE: категориальные признаки без quality (чтобы quality остался внешней меткой)
    # но задание — категориальные признаки; quality тоже можно включить как признак.
    # Для честной внешней оценки quality НЕ включаем в признаки.
    feats = df.drop(columns=["quality"])
    cat_df = discretize_features(feats, n_bins=3)
    cat_df.to_csv(RES / "categorical_features.csv", encoding="utf-8-sig", index=False)

    # частоты категорий
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.ravel()
    for i, col in enumerate(cat_df.columns):
        vc = cat_df[col].value_counts()
        axes[i].bar(vc.index.astype(str), vc.values, color="#74a9cf")
        axes[i].set_title(FEATURE_RU.get(col, col))
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Распределение категорий после дискретизации (терцили)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "04b_categorical_bars.png")
    plt.close(fig)

    transactions = to_transactions(cat_df)
    y_true = df["quality"].values

    print("Running CLOPE...")
    metrics_clope, labels_clope, _ = run_clope(transactions, y_true, df, cat_df)
    print("CLOPE:", metrics_clope["params"], metrics_clope["external"])

    print("Running Gower clustering...")
    metrics_gower, labels_gower, _ = run_gower_clustering(df, cat_df, y_true)
    print("Gower:", metrics_gower["params"], metrics_gower["external"])

    cmp, agree = compare_methods(y_true, labels_clope, labels_gower, metrics_clope, metrics_gower)
    print(cmp)
    print(agree)

    summary = {
        "dataset": {
            "name": "Wine Quality (red)",
            "source": "https://www.kaggle.com/datasets/yasserh/wine-quality-dataset",
            "n_rows": desc_info["n_rows"],
            "n_cols": desc_info["n_cols"],
            "missing": desc_info["missing"],
            "quality_counts": desc_info["quality_counts"],
        },
        "clope": metrics_clope,
        "gower": metrics_gower,
        "comparison": cmp.to_dict(orient="records"),
        "agreement": agree.to_dict(orient="records"),
    }
    with open(RES / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # итоговая таблица назначений
    out = df.copy()
    out["cluster_clope"] = labels_clope
    out["cluster_gower"] = labels_gower
    out.to_csv(RES / "clustered_data.csv", encoding="utf-8-sig", index=False)
    print("Done. Results saved to", RES)


if __name__ == "__main__":
    main()
