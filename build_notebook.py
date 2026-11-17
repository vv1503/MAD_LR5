"""Генерация Jupyter Notebook лабораторной работы по CLOPE."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Лабораторная работа: кластеризация категориальных данных. Алгоритм CLOPE

**Датасет:** [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) (красное вино, UCI / Kaggle).

**Цель:** выполнить кластеризацию категориальных (дискретизированных) признаков методом CLOPE, сравнить с кластеризацией на мере Гауэра (PAM / k-medoids) и методом k-modes, оценить качество и интерпретировать результаты.

## Содержание
1. Импорт библиотек и загрузка данных  
2. Дескриптивный анализ  
3. Преобразование к категориальному / транзакционному виду  
4. Алгоритм CLOPE: теория, реализация, подбор `r`  
5. Кластеризация на мере Гауэра (PAM) и k-modes  
6. Метрики качества, значимость признаков, сравнение  
""")

md("""
## 1. Импорт библиотек

Ниже подключаются библиотеки для анализа данных, визуализации, реализации CLOPE, расчёта меры Гауэра и k-modes. Модуль `clope.py` содержит собственную реализацию алгоритма CLOPE.
""")

code("""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gower import gower_matrix
from kmodes.kmodes import KModes
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder

from clope import CLOPE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(".").resolve()
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
""")

md("""
## 2. Загрузка и описание датасета

Датасет содержит физико-химические характеристики красного вина и экспертную оценку качества `quality` (шкала 3–8).  
Все признаки, кроме `quality`, исходно числовые. Для CLOPE их необходимо дискретизировать в категории.
""")

code("""
def load_wine(path: str = "data/winequality-red.csv") -> pd.DataFrame:
    \"\"\"Загружает CSV Wine Quality (разделитель ';').\"\"\"
    return pd.read_csv(path, sep=";")


df = load_wine()
print(f"Размерность: {df.shape[0]} объектов × {df.shape[1]} признаков")
print(f"Пропуски: {df.isna().sum().sum()}")
df.head()
""")

md("""
### 2.1. Дескриптивная статистика

Рассчитываются среднее, СКО, квартили, асимметрия и эксцесс. Это помогает понять масштаб признаков и наличие выбросов перед дискретизацией.
""")

code("""
def descriptive_table(data: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Строит расширенную дескриптивную таблицу признаков.\"\"\"
    out = data.describe().T
    out["skew"] = data.skew(numeric_only=True)
    out["kurtosis"] = data.kurtosis(numeric_only=True)
    return out


desc = descriptive_table(df)
desc.to_csv(RES / "descriptive_stats.csv", encoding="utf-8-sig")
desc.round(3)
""")

code("""
def plot_histograms(data: pd.DataFrame, save_path: Path) -> None:
    \"\"\"Строит гистограммы всех признаков и сохраняет рисунок.\"\"\"
    cols = list(data.columns)
    nrows = int(np.ceil(len(cols) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 3.2 * nrows))
    axes = axes.ravel()
    for i, col in enumerate(cols):
        axes[i].hist(data[col], bins=20, color="#2c7fb8", edgecolor="white")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Гистограммы признаков Wine Quality", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()


plot_histograms(df, FIG / "01_histograms.png")
""")

code("""
def plot_correlation(data: pd.DataFrame, save_path: Path) -> None:
    \"\"\"Строит тепловую карту корреляций Пирсона.\"\"\"
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, square=True, ax=ax)
    ax.set_title("Корреляционная матрица")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()


plot_correlation(df, FIG / "02_correlation.png")
""")

code("""
def plot_quality_counts(data: pd.DataFrame, save_path: Path) -> None:
    \"\"\"Строит столбчатую диаграмму распределения quality.\"\"\"
    vc = data["quality"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(vc.index.astype(str), vc.values, color="#41b6c4", edgecolor="white")
    ax.set_xlabel("quality")
    ax.set_ylabel("Число наблюдений")
    ax.set_title("Распределение оценки качества")
    for x, y in zip(vc.index.astype(str), vc.values):
        ax.text(x, y + 8, str(y), ha="center")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()
    return vc


quality_counts = plot_quality_counts(df, FIG / "03_quality_distribution.png")
quality_counts
""")

md("""
**Краткие выводы дескриптивного анализа**
- Пропусков нет, выбросы заметны по `chlorides`, `residual sugar`, `sulphates`.
- `alcohol` положительно связан с `quality`, `volatile acidity` — отрицательно.
- Классы `quality` несбалансированы: преобладают оценки 5 и 6.
""")

md("""
## 3. Преобразование данных

CLOPE работает с **транзакциями** — наборами категориальных элементов.  
Числовые признаки дискретизируются по терцилям (`Low` / `Medium` / `High`).  
Признак `quality` **не включается** в кластеризацию и используется как внешняя метка для ARI/NMI.
""")

code("""
def discretize(data: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    \"\"\"Дискретизирует числовые столбцы на n_bins квантильных категорий.\"\"\"
    labels = ["Low", "Medium", "High"][:n_bins]
    out = pd.DataFrame(index=data.index)
    for col in data.columns:
        try:
            out[col] = pd.qcut(data[col], q=n_bins, labels=labels, duplicates="drop")
        except ValueError:
            out[col] = pd.cut(data[col], bins=n_bins, labels=labels, include_lowest=True)
        out[col] = out[col].astype(str)
    return out


def to_transactions(cat_df: pd.DataFrame) -> list:
    \"\"\"Превращает строки таблицы в транзакции вида 'признак=категория'.\"\"\"
    return [[f"{c}={row[c]}" for c in cat_df.columns] for _, row in cat_df.iterrows()]


features = df.drop(columns=["quality"])
cat_df = discretize(features, n_bins=3)
transactions = to_transactions(cat_df)
cat_df.to_csv(RES / "categorical_features.csv", encoding="utf-8-sig", index=False)

print("Пример транзакции:")
print(transactions[0])
cat_df.head()
""")

md("""
## 4. Алгоритм CLOPE

### 4.1. Идея метода

CLOPE (Clustering with SLOPE) — алгоритм кластеризации **категориальных / транзакционных** данных (Yang, Guan, Yu, 2002).  
Для кластера $C$ строится гистограмма частот элементов:

- $S(C)$ — **площадь** (сумма частот всех элементов),  
- $W(C)$ — **ширина** (число различных элементов),  
- $H(C)=S(C)/W(C)$ — средняя высота,  
- $N=|C|$ — число транзакций.

Глобальный критерий (profit):

$$
Profit(C,r)=\\frac{\\sum_{i=1}^{k}\\dfrac{S(C_i)}{W(C_i)^r}\\cdot |C_i|}{\\sum_{i=1}^{k}|C_i|}
$$

Параметр **repulsion** $r>0$ регулирует «жёсткость» сходства: чем больше $r$, тем больше кластеров и тем сильнее требование к совпадению элементов.

Приращение при добавлении транзакции $t$ в кластер:

$$
\\Delta_{add}(C,t,r)=\\frac{S_{new}\\cdot(N+1)}{W_{new}^r}-\\frac{S\\cdot N}{W^r}
$$

Транзакция помещается в кластер с максимальным $\\Delta_{add}$ (либо образует новый).
""")

code("""
# Реализация CLOPE находится в clope.py:
# - Cluster.add / remove — обновление S, W, N и словаря Occ
# - Cluster.delta_add — расчёт Δ_add
# - CLOPE.fit — фаза инициализации + итеративное улучшение Profit

y_true = df["quality"].to_numpy()
""")

md("""
### 4.2. Подбор коэффициента отталкивания $r$

Перебираем $r\\in\\{1.5,1.8,2.0,2.2,2.5,2.8,3.0\\}$.  
Критерии выбора: внешняя метрика NMI относительно `quality`, число кластеров 2–6 и отсутствие вырожденных кластеров.
""")

code("""
def encode_categorical(cat: pd.DataFrame) -> np.ndarray:
    \"\"\"Кодирует категории целыми метками для расстояния Хэмминга.\"\"\"
    enc = cat.copy()
    for col in enc.columns:
        enc[col] = LabelEncoder().fit_transform(enc[col].astype(str))
    return enc.to_numpy()


def hamming_matrix(X: np.ndarray) -> np.ndarray:
    \"\"\"Считает матрицу долей несовпадающих признаков (Хэмминг).\"\"\"
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = (X[i] != X).mean(axis=1)
    return D


def external_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    \"\"\"Считает ARI, NMI, Homogeneity, Completeness, V-measure.\"\"\"
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "Homogeneity": float(homogeneity_score(y_true, y_pred)),
        "Completeness": float(completeness_score(y_true, y_pred)),
        "V-measure": float(v_measure_score(y_true, y_pred)),
    }


X_enc = encode_categorical(cat_df)
rng = np.random.default_rng(42)
idx = rng.choice(len(transactions), size=min(800, len(transactions)), replace=False)
D_sub = hamming_matrix(X_enc[idx])

grid_r = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]
rows = []
best = None
best_score = -1.0

for r in grid_r:
    model = CLOPE(repulsion=r, max_iter=15)
    labels = np.array(model.fit_predict(transactions))
    k = len(np.unique(labels))
    ext = external_metrics(y_true, labels)
    sizes = pd.Series(labels).value_counts()
    min_share = float(sizes.min() / len(labels))
    sil = np.nan
    if 1 < k < len(idx):
        try:
            sil = float(silhouette_score(D_sub, labels[idx], metric="precomputed"))
        except Exception:
            sil = np.nan
    rows.append({"r": r, "n_clusters": k, "profit": model.profit_history_[-1],
                 "silhouette_hamming": sil, "min_cluster_share": min_share, **ext})
    score = ext["NMI"] + 0.15 * (0.0 if np.isnan(sil) else sil)
    if 2 <= k <= 6 and min_share >= 0.05 and score > best_score:
        best_score = score
        best = (r, labels, model)

clope_grid = pd.DataFrame(rows)
clope_grid.to_csv(RES / "clope_param_grid.csv", encoding="utf-8-sig", index=False)
r_best, labels_clope, model_clope = best
print(f"Выбран r={r_best}, кластеров={len(np.unique(labels_clope))}")
clope_grid.round(4)
""")

code("""
def plot_clope_grid(grid: pd.DataFrame, save_path: Path) -> None:
    \"\"\"Визуализирует зависимость числа кластеров и NMI от r.\"\"\"
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(grid["r"], grid["n_clusters"], "o-", color="#2c7fb8")
    ax1.set_xlabel("r (repulsion)")
    ax1.set_ylabel("Число кластеров", color="#2c7fb8")
    ax2 = ax1.twinx()
    ax2.plot(grid["r"], grid["NMI"], "s--", color="#d95f0e")
    ax2.set_ylabel("NMI vs quality", color="#d95f0e")
    ax1.set_title("Подбор параметра repulsion (CLOPE)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()


plot_clope_grid(clope_grid, FIG / "05_clope_param_selection.png")
""")

code("""
def cluster_profile(data: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    \"\"\"Считает средние исходных признаков и размер каждого кластера.\"\"\"
    tmp = data.copy()
    tmp["cluster"] = labels
    out = tmp.groupby("cluster").mean(numeric_only=True)
    out["size"] = tmp.groupby("cluster").size()
    return out


def feature_anova(data: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    \"\"\"Оценивает значимость признаков ANOVA F относительно кластеров.\"\"\"
    X = data.select_dtypes(include=[np.number])
    F, p = f_classif(X, labels)
    out = pd.DataFrame({"feature": X.columns, "F": F, "p_value": p}).sort_values("F", ascending=False)
    return out


profile_clope = cluster_profile(df, labels_clope)
sig_clope = feature_anova(df, labels_clope)
metrics_clope = external_metrics(y_true, labels_clope)

print("Метрики CLOPE:", metrics_clope)
print("Profit =", round(model_clope.profit_history_[-1], 4))
print(profile_clope.round(3))
print(sig_clope.head(8))
""")

code("""
def plot_quality_by_cluster(labels: np.ndarray, quality: np.ndarray, title: str, save_path: Path) -> None:
    \"\"\"Строит stacked bar состава quality внутри кластеров.\"\"\"
    ct = pd.crosstab(labels, quality, normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_xlabel("Кластер")
    ax.set_ylabel("Доля, %")
    ax.set_title(title)
    ax.legend(title="quality", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()


plot_quality_by_cluster(labels_clope, y_true, f"Состав кластеров CLOPE (r={r_best})",
                        FIG / "quality_by_cluster_clope.png")
""")

md("""
### 4.3. Интерпретация CLOPE

Типичное решение при выбранном $r$: два кластера, разделённые в первую очередь по **алкоголю** и **плотности**.
- Кластер с более высоким `alcohol` имеет в среднем более высокое `quality`.
- Кластер с более низким `alcohol` / более высокой плотностью — более «рядовые» вина.

Это согласуется с корреляционным анализом.
""")

md("""
## 5. Кластеризация на мере Гауэра и метод k-modes

### 5.1. Мера Гауэра

Для объектов $i,j$ и признаков $f=1..p$:

$$
d_{Gower}(i,j)=\\frac{\\sum_{f=1}^{p}w_{ijf}\\delta_{ijf}}{\\sum_{f=1}^{p}w_{ijf}}
$$

где для числового признака $\\delta_{ijf}=|x_{if}-x_{jf}|/R_f$, для категориального $\\delta_{ijf}=0$ при совпадении и $1$ иначе.

Далее применяется **PAM (k-medoids)** по матрице Гауэра: центры кластеров — реальные объекты (медоиды).

### 5.2. K-modes

K-modes минимизирует сумму несходств Хэмминга до мод кластеров и подходит для чисто категориальных данных (наш дискретизированный набор).
""")

code("""
def pam_gower(Gw: np.ndarray, k: int, n_init: int = 10, max_iter: int = 30, seed: int = 42) -> np.ndarray:
    \"\"\"Выполняет PAM (k-medoids) по предвычисленной матрице расстояний.\"\"\"
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


Gw = gower_matrix(features.to_numpy())
gower_rows = []
best_g = None
best_g_score = -1.0

for k in range(2, 7):
    labels = pam_gower(Gw, k=k)
    sil = float(silhouette_score(Gw, labels, metric="precomputed"))
    ext = external_metrics(y_true, labels)
    min_share = float(pd.Series(labels).value_counts().min() / len(labels))
    gower_rows.append({"k": k, "silhouette_gower": sil, "min_cluster_share": min_share, **ext})
    score = sil + 0.2 * ext["NMI"]
    if min_share >= 0.05 and score > best_g_score:
        best_g_score = score
        best_g = (k, labels, sil)

gower_grid = pd.DataFrame(gower_rows)
gower_grid.to_csv(RES / "gower_param_grid.csv", encoding="utf-8-sig", index=False)
k_gower, labels_gower, sil_gower = best_g
print(f"Gower+PAM: k={k_gower}, silhouette={sil_gower:.4f}")
gower_grid.round(4)
""")

code("""
profile_gower = cluster_profile(df, labels_gower)
sig_gower = feature_anova(df, labels_gower)
metrics_gower = external_metrics(y_true, labels_gower)
print("Метрики Gower+PAM:", metrics_gower)
print(profile_gower.round(3))
plot_quality_by_cluster(labels_gower, y_true, f"Состав кластеров Gower+PAM (k={k_gower})",
                        FIG / "quality_by_cluster_gower.png")
""")

code("""
km_rows = []
best_km = None
best_km_score = -1.0
for k in range(2, 7):
    km = KModes(n_clusters=k, init="Huang", n_init=8, verbose=0, random_state=42)
    labels = km.fit_predict(X_enc)
    ext = external_metrics(y_true, labels)
    km_rows.append({"k": k, **ext})
    if ext["NMI"] > best_km_score:
        best_km_score = ext["NMI"]
        best_km = (k, labels)

kmodes_grid = pd.DataFrame(km_rows)
k_km, labels_kmodes = best_km
metrics_kmodes = external_metrics(y_true, labels_kmodes)
profile_kmodes = cluster_profile(df, labels_kmodes)
print(f"K-modes: k={k_km}")
print(metrics_kmodes)
kmodes_grid.round(4)
""")

md("""
## 6. Сравнение методов

Используются **внешние** метрики относительно `quality`:
- **ARI** — скорректированный индекс Рэнда (случайное согласование ≈ 0);
- **NMI** — нормализованная взаимная информация;
- **Homogeneity / Completeness / V-measure**.

Внутренние метрики: silhouette по Хэммингу (для категориальных) и по Гауэру (для PAM).
""")

code("""
cmp = pd.DataFrame([
    {"method": "CLOPE", "n_clusters": len(np.unique(labels_clope)), **metrics_clope},
    {"method": "Gower+PAM", "n_clusters": len(np.unique(labels_gower)), **metrics_gower},
    {"method": "K-modes", "n_clusters": len(np.unique(labels_kmodes)), **metrics_kmodes},
])
cmp.to_csv(RES / "comparison_metrics.csv", encoding="utf-8-sig", index=False)

agree = pd.DataFrame({
    "pair": ["CLOPE vs Gower+PAM", "CLOPE vs quality", "Gower+PAM vs quality"],
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
ax.set_ylabel("Значение")
ax.set_title("Сравнение методов (внешние метрики vs quality)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "10_comparison.png")
plt.show()

print(cmp.round(4))
print(agree.round(4))
""")

code("""
out = df.copy()
out["cluster_clope"] = labels_clope
out["cluster_gower"] = labels_gower
out["cluster_kmodes"] = labels_kmodes
out.to_csv(RES / "clustered_data.csv", encoding="utf-8-sig", index=False)
print("Результаты сохранены в папки results/ и figures/")
""")

md("""
## 7. Итоговые выводы

1. Для CLOPE числовые признаки красного вина успешно приведены к транзакционному виду; параметр $r$ управляет детализацией разбиения.
2. Лучшее интерпретируемое CLOPE-решение разделяет вина главным образом по алкоголю/плотности и частично согласуется с `quality`.
3. Gower + PAM работает с исходными числовыми шкалами без жёсткой дискретизации и даёт иное, более «кислотно-структурное» разбиение.
4. K-modes на категориях даёт более дробную структуру; согласование с `quality` умеренное.
5. Методы выявляют разные оси неоднородности данных: CLOPE — «крепость/качество», Gower — кислотный профиль и тело вина. Это нормально для разных функций сходства.

## Краткие ответы к защите
- **Смешанные данные:** k-prototypes, Gower + PAM/иерархия, FAMD + k-means.  
- **Категориальные:** k-modes, CLOPE, ROCK, COOLCAT, Hamming + hierarchical.  
- **K-modes:** моды вместо средних, расстояние Хэмминга.  
- **K-prototypes:** смесь Euclidean (числовые) и Hamming (категории).  
- **Меры:** Хэмминг, Журавлев, Гауэр, Jaccard для множеств/транзакций.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
out = ROOT / "LR5_CLOPE_WineQuality.ipynb"
nbf.write(nb, out)
print("Notebook written:", out)
