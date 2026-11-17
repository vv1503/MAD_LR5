"""Генерация подробного Word-отчёта по лабораторной работе CLOPE."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, Inches, RGBColor

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"


def set_run_font(run, name="Times New Roman", size=14, bold=False, italic=False):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_para(doc, text, *, size=14, bold=False, italic=False, align="justify", space_after=6, first_line=True):
    p = doc.add_paragraph()
    if align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "left":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if first_line and align == "justify":
        pf.first_line_indent = Cm(1.25)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_heading_custom(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run_font(run, size=16 if level == 1 else 14, bold=True)
    return h


def add_formula(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_run_font(run, size=13, italic=True)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run(text)
    set_run_font(run, size=12, italic=True)
    return p


def add_picture(doc, path: Path, width_cm=15.5):
    if not path.exists():
        add_para(doc, f"[Рисунок не найден: {path.name}]", italic=True, first_line=False)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))


def add_table_from_df(doc, df: pd.DataFrame, max_rows=20, float_round=3):
    show = df.copy()
    if len(show) > max_rows:
        show = show.head(max_rows)
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: f"{x:.{float_round}f}" if pd.notna(x) else "")
        else:
            show[c] = show[c].astype(str)

    table = doc.add_table(rows=1, cols=len(show.columns))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, col in enumerate(show.columns):
        hdr[i].text = str(col)
        for p in hdr[i].paragraphs:
            for r in p.runs:
                set_run_font(r, size=11, bold=True)

    for _, row in show.iterrows():
        cells = table.add_row().cells
        for i, col in enumerate(show.columns):
            cells[i].text = str(row[col])
            for p in cells[i].paragraphs:
                for r in p.runs:
                    set_run_font(r, size=11)
    doc.add_paragraph()


def fmt_metrics(d: dict) -> str:
    return ", ".join(f"{k} = {v:.4f}" for k, v in d.items())


def build():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    desc = pd.read_csv(RES / "descriptive_stats.csv", index_col=0)
    clope_grid = pd.read_csv(RES / "clope_param_grid.csv")
    gower_grid = pd.read_csv(RES / "gower_param_grid.csv")
    cmp = pd.read_csv(RES / "comparison_metrics.csv")
    agree = pd.read_csv(RES / "methods_agreement.csv")
    profile_clope = pd.read_csv(RES / "profile_clope.csv")
    profile_gower = pd.read_csv(RES / "profile_gower.csv")
    profile_kmodes = pd.read_csv(RES / "profile_kmodes.csv")
    sig_clope = pd.read_csv(RES / "feature_significance_clope.csv")
    sig_gower = pd.read_csv(RES / "feature_significance_gower.csv")
    kmodes_grid = pd.read_csv(RES / "kmodes_param_grid.csv") if (RES / "kmodes_param_grid.csv").exists() else None

    clope = summary["clope"]
    gower = summary["gower"]

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(3)
    section.right_margin = Cm(1.5)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(14)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    # ---------- Титульный лист ----------
    for _ in range(2):
        doc.add_paragraph()
    add_para(doc, "МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РФ", align="center", bold=True, first_line=False, space_after=3)
    add_para(doc, "Федеральное государственное бюджетное образовательное учреждение высшего образования", align="center", size=12, first_line=False, space_after=3)
    add_para(doc, "[Наименование вуза]", align="center", first_line=False, space_after=20)

    add_para(doc, "Кафедра [наименование кафедры]", align="center", first_line=False, space_after=40)

    add_para(doc, "ОТЧЁТ", align="center", bold=True, size=16, first_line=False, space_after=6)
    add_para(doc, "по лабораторной работе", align="center", bold=True, first_line=False, space_after=6)
    add_para(doc, "«Кластеризация категориальных данных. Алгоритм CLOPE»", align="center", bold=True, size=14, first_line=False, space_after=6)
    add_para(doc, "Дисциплина: Методы анализа данных", align="center", first_line=False, space_after=40)

    add_para(doc, "Выполнил(а): студент(ка) группы ________", align="right", first_line=False, space_after=3)
    add_para(doc, "________________ / ФИО /", align="right", first_line=False, space_after=12)
    add_para(doc, "Проверил(а): ________________", align="right", first_line=False, space_after=3)
    add_para(doc, "________________ / ФИО /", align="right", first_line=False, space_after=40)

    add_para(doc, "2026", align="center", first_line=False)
    doc.add_page_break()

    # ---------- Цель ----------
    add_heading_custom(doc, "Цель работы", 1)
    add_para(
        doc,
        "Изучить метод кластеризации категориальных (транзакционных) данных CLOPE, "
        "выполнить дескриптивный анализ набора Wine Quality, преобразовать числовые признаки "
        "в категориальные, получить кластерную структуру CLOPE, сравнить её с результатами "
        "кластеризации на мере расстояния Гауэра (алгоритм PAM / k-medoids) и методом k-modes, "
        "оценить качество кластеризации и интерпретировать решение."
    )

    add_heading_custom(doc, "1. Описание датасета", 1)
    add_para(
        doc,
        "В работе использован набор данных Wine Quality (красное вино), доступный на Kaggle "
        "(https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) и в репозитории UCI. "
        f"Массив содержит {summary['dataset']['n_rows']} наблюдений и {summary['dataset']['n_cols']} признаков. "
        f"Пропущенных значений: {summary['dataset']['missing']}."
    )
    add_para(
        doc,
        "Признаки описывают физико-химический состав вина: fixed acidity (фиксированная кислотность), "
        "volatile acidity (летучая кислотность), citric acid (лимонная кислота), residual sugar "
        "(остаточный сахар), chlorides (хлориды), free sulfur dioxide и total sulfur dioxide "
        "(свободный и общий диоксид серы), density (плотность), pH, sulphates (сульфаты), alcohol "
        "(объёмная доля спирта). Целевая переменная quality — экспертная оценка качества по шкале "
        "от 3 до 8. Для внешней оценки качества кластеризации quality используется как эталонная метка "
        "и не включается в признаки алгоритмов кластеризации."
    )
    add_para(
        doc,
        "Распределение оценок качества: "
        + ", ".join(f"quality={k}: {v}" for k, v in summary["dataset"]["quality_counts"].items())
        + ". Наблюдается дисбаланс классов: преобладают оценки 5 и 6."
    )

    add_heading_custom(doc, "2. Дескриптивный анализ переменных", 1)
    add_para(
        doc,
        "Для каждого признака рассчитаны число наблюдений, среднее, стандартное отклонение, "
        "минимум, квартили, максимум, а также коэффициенты асимметрии (skewness) и эксцесса (kurtosis). "
        "Асимметрия характеризует смещение распределения относительно среднего, эксцесс — тяжесть хвостов "
        "и островершинность относительно нормального закона."
    )
    add_para(doc, "Таблица 1 – Фрагмент дескриптивной статистики", align="center", italic=True, first_line=False, size=12)
    show_desc = desc[["mean", "std", "min", "50%", "max", "skew", "kurtosis"]].copy()
    show_desc.insert(0, "feature", show_desc.index)
    show_desc = show_desc.reset_index(drop=True)
    add_table_from_df(doc, show_desc, float_round=3)

    add_para(
        doc,
        "На гистограммах (рисунок 1) видно, что ряд признаков (chlorides, residual sugar, sulphates) "
        "имеет правостороннюю асимметрию и выбросы. Признак alcohol близок к умеренно скошенному "
        "распределению. Оценка quality принимает дискретные значения с модой в области 5–6."
    )
    add_picture(doc, FIG / "01_histograms.png")
    add_caption(doc, "Рисунок 1 – Гистограммы признаков набора Wine Quality")

    add_para(
        doc,
        "Корреляционный анализ (рисунок 2) показывает положительную связь alcohol и quality, "
        "а также отрицательную связь volatile acidity и quality. Плотность положительно связана "
        "с fixed acidity и отрицательно — с alcohol. Эти зависимости важны для последующей "
        "интерпретации кластеров."
    )
    add_picture(doc, FIG / "02_correlation.png", width_cm=14)
    add_caption(doc, "Рисунок 2 – Корреляционная матрица признаков (коэффициент Пирсона)")

    add_picture(doc, FIG / "03_quality_distribution.png", width_cm=12)
    add_caption(doc, "Рисунок 3 – Распределение экспертной оценки quality")

    add_picture(doc, FIG / "04_boxplots.png")
    add_caption(doc, "Рисунок 4 – Диаграммы размаха признаков (оценка выбросов)")

    add_para(
        doc,
        "Вывод по дескриптивному анализу: данные пригодны для кластеризации; пропусков нет; "
        "масштабы признаков различны, поэтому для CLOPE необходима категоризация, а для меры Гауэра — "
        "встроенная нормировка по размаху каждого признака. Наличие выбросов учитывается при интерпретации."
    )

    add_heading_custom(doc, "3. Преобразование данных", 1)
    add_para(
        doc,
        "Алгоритм CLOPE оперирует транзакциями — множествами категориальных элементов. "
        "Поэтому все числовые признаки (кроме quality) были дискретизированы по терцилям "
        "выборочного распределения: категории Low, Medium, High. При невозможности построить "
        "ровно три квантильные корзины из-за повторяющихся значений применялось равномерное разбиение "
        "диапазона (pd.cut)."
    )
    add_para(
        doc,
        "Каждая строка таблицы преобразуется в транзакцию вида "
        "{fixed acidity=Low, volatile acidity=High, …, alcohol=Medium}. "
        "Такое представление совместимо с логикой CLOPE и с демо-примером Loginom "
        "(кластеризация транзакций)."
    )
    add_picture(doc, FIG / "04b_categorical_bars.png")
    add_caption(doc, "Рисунок 5 – Распределения категорий после дискретизации признаков")

    add_heading_custom(doc, "4. Теоретические сведения о методах", 1)

    add_heading_custom(doc, "4.1. Кластеризация категориальных и смешанных данных", 2)
    add_para(
        doc,
        "Классический алгоритм k-средних опирается на евклидово расстояние и средние значения, "
        "поэтому напрямую плохо применим к категориальным признакам. Для категориальных данных "
        "используют k-modes (моды вместо средних, расстояние Хэмминга), CLOPE, ROCK, COOLCAT и др. "
        "Для смешанных данных применяют k-prototypes, меры Гауэра / Журавлева с последующим "
        "PAM или иерархической кластеризацией, а также факторизацию смешанных данных (FAMD) "
        "с последующим k-means в латентном пространстве."
    )

    add_heading_custom(doc, "4.2. Алгоритм CLOPE", 2)
    add_para(
        doc,
        "CLOPE (Clustering with SLOPE; Yang, Guan, Yu, 2002) максимизирует глобальный критерий "
        "Profit, основанный на форме гистограммы элементов внутри кластера. Для кластера C "
        "обозначим: Occ(i, C) — число вхождений элемента i; S(C) = Σ Occ(i, C) — площадь; "
        "W(C) — число различных элементов (ширина); N = |C| — число транзакций; "
        "H(C) = S(C)/W(C) — средняя высота гистограммы."
    )
    add_para(doc, "Градиент (наклон) кластера с параметром отталкивания r:")
    add_formula(doc, "G(C, r) = S(C) / W(C)^r")
    add_para(doc, "Глобальный критерий:")
    add_formula(doc, "Profit(C, r) = [ Σ_i G(C_i, r) · |C_i| ] / [ Σ_i |C_i| ]")
    add_para(
        doc,
        "Чем выше отношение высоты к ширине, тем больше «перекрытие» элементов транзакций "
        "внутри кластера. Параметр r (repulsion) усиливает штраф за ширину: при увеличении r "
        "алгоритм предпочитает более однородные и, как правило, более многочисленные кластеры."
    )
    add_para(doc, "Для эффективного поиска используется приращение при добавлении транзакции t:")
    add_formula(doc, "Δ_add(C, t, r) = S_new·(N+1)/W_new^r − S·N/W^r")
    add_para(
        doc,
        "где S_new = S + |t|, а W_new = W + число новых элементов транзакции, отсутствующих в C. "
        "Теорема авторов: выбор кластера с максимальным Δ_add максимизирует Profit. "
        "Алгоритм включает фазу инициализации (последовательное назначение транзакций) и "
        "итеративную фазу перемещений до стабилизации разбиения."
    )

    add_heading_custom(doc, "4.3. K-modes и k-prototypes", 2)
    add_para(
        doc,
        "K-modes (Huang, 1998) минимизирует сумму расстояний Хэмминга от объектов до мод кластеров. "
        "Мода категориального признака — наиболее частое значение в кластере. Обновление мод и "
        "переназначение объектов чередуются до сходимости."
    )
    add_para(
        doc,
        "K-prototypes объединяет евклидову компоненту для числовых признаков и компоненту "
        "несходства (Хэмминг) для категориальных. Целевая функция:"
    )
    add_formula(doc, "E = Σ_l Σ_i u_il [ Σ_j∈num (x_ij − q_lj)^2 + γ Σ_j∈cat δ(x_ij, q_lj) ]")
    add_para(
        doc,
        "где u_il — индикатор принадлежности, q_lj — центр (среднее или мода), γ — вес категориальной части, "
        "δ — функция совпадения/несовпадения категорий."
    )

    add_heading_custom(doc, "4.4. Меры расстояния", 2)
    add_para(doc, "Расстояние Хэмминга для p категориальных признаков:")
    add_formula(doc, "d_H(x, y) = (1/p) · Σ_j 1[x_j ≠ y_j]")
    add_para(doc, "Мера Гауэра для смешанных данных:")
    add_formula(doc, "d_Gower(i, j) = Σ_f w_ijf · δ_ijf  /  Σ_f w_ijf")
    add_para(
        doc,
        "Для числового признака δ_ijf = |x_if − x_jf| / R_f, где R_f — размах признака f. "
        "Для категориального признака δ_ijf = 0 при равенстве значений и 1 иначе. "
        "Вес w_ijf = 0, если сравнение по признаку невозможно (пропуск)."
    )
    add_para(
        doc,
        "Мера Журавлева основана на покомпонентном сравнении с учётом информативности признаков "
        "и часто используется в распознавании смешанных описаний. В данной работе в качестве "
        "основной альтернативы CLOPE выбрана мера Гауэра, рекомендуемая заданием (в т.ч. в Jamovi)."
    )

    add_heading_custom(doc, "4.5. Метрики качества кластеризации", 2)
    add_para(
        doc,
        "Поскольку имеется внешняя метка quality, рассчитаны внешние метрики. "
        "Adjusted Rand Index (ARI) корректирует индекс Рэнда на случайное согласование; "
        "значение близко к 0 при случайном разбиении и к 1 при полном совпадении."
    )
    add_formula(doc, "ARI = (RI − E[RI]) / (max(RI) − E[RI])")
    add_para(doc, "Нормализованная взаимная информация:")
    add_formula(doc, "NMI(U, V) = I(U, V) / sqrt(H(U)·H(V))")
    add_para(
        doc,
        "Homogeneity требует, чтобы каждый кластер содержал объекты одного класса; "
        "Completeness — чтобы объекты одного класса попадали в один кластер; "
        "V-measure — их гармоническое среднее. "
        "Внутренняя метрика silhouette для объекта i:"
    )
    add_formula(doc, "s(i) = (b(i) − a(i)) / max{a(i), b(i)}")
    add_para(
        doc,
        "где a(i) — среднее расстояние до объектов своего кластера, b(i) — минимальное среднее "
        "расстояние до объектов другого кластера. В работе silhouette считался по Хэммингу "
        "(для CLOPE) и по Гауэру (для PAM)."
    )
    add_para(
        doc,
        "Значимость признаков оценивалась однофакторным дисперсионным анализом (ANOVA): "
        "F-статистика сравнивает межкластерную и внутрикластерную дисперсию признака; "
        "малые p-value свидетельствуют о связи признака с кластерной структурой."
    )

    add_heading_custom(doc, "5. Результаты применения метода CLOPE", 1)
    add_para(
        doc,
        f"Реализован алгоритм CLOPE (модуль clope.py). Выполнен перебор параметра repulsion "
        f"r ∈ {{1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0}}. При росте r число кластеров быстро увеличивается "
        f"(от 2 до сотен), что соответствует теории: усиливается требование к однородности транзакций."
    )
    add_para(doc, "Таблица 2 – Подбор параметра repulsion для CLOPE", align="center", italic=True, first_line=False, size=12)
    add_table_from_df(doc, clope_grid, float_round=4)

    add_picture(doc, FIG / "05_clope_param_selection.png", width_cm=14)
    add_caption(doc, "Рисунок 6 – Зависимость числа кластеров и NMI от параметра r")

    r_best = clope["params"]["r"]
    k_clope = clope["params"]["n_clusters"]
    add_para(
        doc,
        f"В качестве рабочего решения выбрано r = {r_best}: число кластеров k = {k_clope}, "
        f"кластеры невырождены по размеру, достигается наилучший баланс интерпретируемости и "
        f"согласованности с quality среди решений с 2–6 кластерами. "
        f"Итоговое значение Profit(C, r) = {clope['profit']:.4f}."
    )
    add_para(doc, "Внешние метрики CLOPE относительно quality: " + fmt_metrics(clope["external"]) + ".")

    add_picture(doc, FIG / "06_clope_profit.png", width_cm=12)
    add_caption(doc, "Рисунок 7 – Динамика критерия Profit по итерациям CLOPE")

    add_picture(doc, FIG / "07_clope_sizes.png", width_cm=12)
    add_caption(doc, f"Рисунок 8 – Размеры кластеров CLOPE (r = {r_best})")

    add_para(doc, "Таблица 3 – Профили кластеров CLOPE (средние исходных признаков)", align="center", italic=True, first_line=False, size=12)
    add_table_from_df(doc, profile_clope.round(3), float_round=3)

    add_picture(doc, FIG / "profiles_clope.png")
    add_caption(doc, "Рисунок 9 – Стандартизированные профили кластеров CLOPE")

    add_picture(doc, FIG / "quality_by_cluster_clope.png", width_cm=13)
    add_caption(doc, "Рисунок 10 – Распределение quality внутри кластеров CLOPE")

    add_picture(doc, FIG / "significance_clope.png", width_cm=13)
    add_caption(doc, "Рисунок 11 – Значимость признаков (ANOVA F) для решения CLOPE")

    add_para(doc, "Таблица 4 – Наиболее значимые признаки для CLOPE", align="center", italic=True, first_line=False, size=12)
    add_table_from_df(doc, sig_clope.head(8), float_round=4)

    # Interpret CLOPE from profiles
    c0 = profile_clope.iloc[0]
    c1 = profile_clope.iloc[1] if len(profile_clope) > 1 else None
    if c1 is not None:
        if c0["alcohol"] < c1["alcohol"]:
            low, high = c0, c1
            i_low, i_high = int(c0.get("cluster", 0)), int(c1.get("cluster", 1))
        else:
            low, high = c1, c0
            i_low, i_high = int(c1.get("cluster", 1)), int(c0.get("cluster", 0))
        add_para(
            doc,
            f"Интерпретация. Кластер {i_low} (n = {int(low['size'])}) характеризуется более низким "
            f"средним содержанием алкоголя ({low['alcohol']:.2f}%) и более низкой средней оценкой "
            f"качества ({low['quality']:.2f}). Кластер {i_high} (n = {int(high['size'])}) объединяет "
            f"вина с более высоким alcohol ({high['alcohol']:.2f}%) и более высоким quality "
            f"({high['quality']:.2f}). Наиболее значимые признаки — alcohol, density, а также "
            f"связь с quality (по ANOVA). Таким образом, CLOPE на дискретизированных признаках "
            f"выделяет «рядовые» и более «крепкие/высокооценённые» вина."
        )

    add_heading_custom(doc, "6. Результаты кластеризации на мере Гауэра и методом k-modes", 1)
    add_para(
        doc,
        "В качестве альтернативы CLOPE использована мера Гауэра на исходных числовых признаках "
        "(без quality) и алгоритм PAM (k-medoids). PAM выбирает медоиды — реальные объекты, "
        "минимизирующие сумму расстояний внутри кластера, что устойчивее к выбросам, чем k-means."
    )
    add_para(doc, "Таблица 5 – Подбор числа кластеров k для PAM по мере Гауэра", align="center", italic=True, first_line=False, size=12)
    add_table_from_df(doc, gower_grid, float_round=4)

    add_picture(doc, FIG / "08_gower_param_selection.png", width_cm=14)
    add_caption(doc, "Рисунок 12 – Подбор k для PAM (silhouette по Гауэру и NMI)")

    k_g = gower["params"]["k"]
    add_para(
        doc,
        f"Выбрано k = {k_g} по максимуму комбинированного критерия (silhouette + вклад NMI) "
        f"при условии, что минимальная доля кластера не менее 5%. "
        f"Silhouette (Gower) = {gower['params']['silhouette']:.4f}. "
        f"Внешние метрики: " + fmt_metrics(gower["external"]) + "."
    )
    add_para(
        doc,
        f"Размеры кластеров: "
        + ", ".join(f"кластер {k}: {v}" for k, v in gower["cluster_sizes"].items())
        + "."
    )

    add_picture(doc, FIG / "09_gower_sizes.png", width_cm=12)
    add_caption(doc, f"Рисунок 13 – Размеры кластеров Gower + PAM (k = {k_g})")

    add_para(doc, "Таблица 6 – Профили кластеров Gower + PAM", align="center", italic=True, first_line=False, size=12)
    add_table_from_df(doc, profile_gower.round(3), float_round=3)

    add_picture(doc, FIG / "profiles_gower.png")
    add_caption(doc, "Рисунок 14 – Стандартизированные профили кластеров (Gower + PAM)")

    add_picture(doc, FIG / "quality_by_cluster_gower.png", width_cm=13)
    add_caption(doc, "Рисунок 15 – Распределение quality внутри кластеров Gower + PAM")

    add_picture(doc, FIG / "significance_gower.png", width_cm=13)
    add_caption(doc, "Рисунок 16 – Значимость признаков для решения Gower + PAM")

    g0 = profile_gower.iloc[0]
    g1 = profile_gower.iloc[1] if len(profile_gower) > 1 else None
    if g1 is not None:
        add_para(
            doc,
            f"Интерпретация Gower + PAM. Кластеры различаются прежде всего кислотным профилем и "
            f"«телом» вина: один кластер имеет более высокую фиксированную кислотность "
            f"(fixed acidity ≈ {max(g0['fixed acidity'], g1['fixed acidity']):.2f}) и более низкую "
            f"летучую кислотность, другой — противоположный профиль. Разделение по alcohol выражено "
            f"слабее, чем у CLOPE. Это ожидаемо: мера Гауэра учитывает все признаки в сопоставимой "
            f"шкале [0, 1], тогда как CLOPE на терцилях сильнее реагирует на совпадение редких "
            f"комбинаций категорий, связанных с крепостью."
        )

    add_para(
        doc,
        "Дополнительно выполнен метод k-modes на дискретизированных признаках. "
        f"По максимуму NMI выбрано k = {gower['kmodes_k']}. "
        f"Метрики k-modes: " + fmt_metrics(gower["kmodes_external"]) + "."
    )
    if kmodes_grid is not None:
        add_para(doc, "Таблица 7 – Подбор k для k-modes", align="center", italic=True, first_line=False, size=12)
        add_table_from_df(doc, kmodes_grid, float_round=4)

    add_para(doc, "Таблица 8 – Профили кластеров k-modes", align="center", italic=True, first_line=False, size=12)
    add_table_from_df(doc, profile_kmodes.round(3), float_round=3)

    add_picture(doc, FIG / "quality_by_cluster_kmodes.png", width_cm=13)
    add_caption(doc, "Рисунок 17 – Распределение quality внутри кластеров k-modes")

    add_heading_custom(doc, "7. Сравнительный анализ полученных решений", 1)
    add_para(doc, "Таблица 9 – Сравнение методов по внешним метрикам (эталон — quality)", align="center", italic=True, first_line=False, size=12)
    add_table_from_df(doc, cmp.round(4), float_round=4)

    add_para(doc, "Таблица 10 – Согласованность разбиений (ARI / NMI)", align="center", italic=True, first_line=False, size=12)
    add_table_from_df(doc, agree.round(4), float_round=4)

    add_picture(doc, FIG / "10_comparison.png", width_cm=14)
    add_caption(doc, "Рисунок 18 – Столбчатая диаграмма сравнения ARI, NMI и V-measure")

    add_para(
        doc,
        "Сравнительный анализ показывает, что CLOPE на категориальных транзакциях даёт более "
        "высокое согласование с экспертной оценкой quality, чем PAM по мере Гауэра и чем k-modes "
        "в выбранных конфигурациях. Вместе с тем ARI/NMI для всех методов остаются умеренными: "
        "quality — субъективная дискретная оценка, а кластеризация выполняется без этой метки и "
        "отражает прежде всего физико-химическую неоднородность выборки."
    )
    add_para(
        doc,
        "Согласованность между CLOPE и Gower+PAM невелика: методы «видят» разные оси вариации. "
        "CLOPE (после квантильной дискретизации) акцентирует сочетания категорий, связанные с "
        "алкоголем и плотностью. Gower+PAM сохраняет метрическую информацию всех признаков и "
        "выделяет кислотно-структурные группы. K-modes даёт более дробное категориальное разбиение "
        "и занимает промежуточное положение по ряду метрик."
    )
    add_para(
        doc,
        "Практическая рекомендация: для интерпретируемого сегментирования вин по «стилю крепости» "
        "удобен CLOPE; для сегментации по полному химическому профилю в исходных шкалах — "
        "мера Гауэра с PAM/медоидами (аналог подхода Jamovi с расстоянием Gower)."
    )

    add_heading_custom(doc, "8. Ответы на вопросы к защите (краткий конспект)", 1)
    add_para(
        doc,
        "1) Методы для категориальных данных: k-modes, CLOPE, ROCK, COOLCAT, иерархия с Хэммингом/Jaccard. "
        "Для смешанных: k-prototypes, Gower/Журавлев + PAM/иерархия, FAMD + k-means."
    )
    add_para(
        doc,
        "2) K-prototypes — обобщение k-means/k-modes: числовая часть через квадрат евклидова расстояния, "
        "категориальная — через несходство; центр кластера = прототип (средние + моды)."
    )
    add_para(
        doc,
        "3) K-modes — только категории; функция стоимости = сумма Хэмминговых расстояний до мод; "
        "итеративное переназначение объектов и пересчёт мод."
    )
    add_para(
        doc,
        "4) CLOPE — транзакционный алгоритм, максимизирует Profit = среднее от S/W^r по кластерам; "
        "не требует матрицы попарных расстояний; параметр r управляет числом кластеров."
    )
    add_para(
        doc,
        "5) Меры: Хэмминг — доля несовпадающих категорий; Гауэр — нормированная смесь числовых и "
        "категориальных несходств; Журавлев — покомпонентное сравнение с весами/информативностью; "
        "для множеств также Jaccard."
    )

    add_heading_custom(doc, "9. Заключение", 1)
    add_para(
        doc,
        "В лабораторной работе выполнен полный цикл анализа: описание и дескриптивная статистика "
        "Wine Quality, преобразование к категориальному/транзакционному виду, кластеризация CLOPE "
        "с подбором repulsion, альтернативная кластеризация на мере Гауэра (PAM) и k-modes, "
        "расчёт внешних и внутренних метрик, оценка значимости признаков и сравнительная интерпретация. "
        "Полученные результаты соответствуют теоретическим свойствам методов и пригодны для защиты."
    )

    add_heading_custom(doc, "Список источников", 1)
    sources = [
        "Yang Y., Guan X., You J. CLOPE: A fast and effective clustering algorithm for transactional data // KDD. 2002.",
        "Huang Z. Extensions to the k-means algorithm for clustering large data sets with categorical values // Data Mining and Knowledge Discovery. 1998.",
        "Gower J.C. A general coefficient of similarity and some of its properties // Biometrics. 1971.",
        "Loginom: Кластеризация транзакций (CLOPE). https://loginom.ru/blog/clope",
        "Loginom Help: Transaction clustering. https://help.loginom.ru/userguide/processors/datamining/transaction-clustering.html",
        "Wine Quality Dataset (Kaggle). https://www.kaggle.com/datasets/yasserh/wine-quality-dataset",
        "Cortez et al. Modeling wine preferences by data mining from physicochemical properties // Decision Support Systems. 2009.",
    ]
    for i, s in enumerate(sources, 1):
        add_para(doc, f"{i}. {s}", first_line=False, space_after=4)

    add_heading_custom(doc, "Приложение. Состав программных материалов", 1)
    add_para(doc, "1. LR5_CLOPE_WineQuality.ipynb — подробный Jupyter Notebook с кодом и пояснениями.", first_line=False)
    add_para(doc, "2. clope.py — реализация алгоритма CLOPE.", first_line=False)
    add_para(doc, "3. analysis_pipeline.py — воспроизводимый пайплайн расчётов и рисунков.", first_line=False)
    add_para(doc, "4. data/winequality-red.csv — исходные данные.", first_line=False)
    add_para(doc, "5. figures/ — рисунки отчёта; results/ — таблицы метрик и метки кластеров.", first_line=False)

    out = ROOT / "Отчет_ЛР5_CLOPE_WineQuality.docx"
    doc.save(out)
    print("Report saved:", out)


if __name__ == "__main__":
    build()
