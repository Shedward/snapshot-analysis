import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium", app_title="Snapshot тестирование")


@app.cell
def _():
    import os
    import marimo as mo
    import pandas as pd
    import PIL as pl
    import numpy as np
    import itertools
    return itertools, mo, np, os, pd, pl


@app.cell
def _(mo):
    mo.md("""# Snapshot тесты""")
    return


@app.cell
def _():
    DATA_DIR = "Data"
    return (DATA_DIR,)


@app.cell
def _(DATA_DIR, os, pd):
    # Загружаем собранные снапшоты

    def dirs(path):
        dirs = os.listdir(path)
        return [d for d in dirs if not d.startswith(".")]

    def load_snapshots(data_dir):
        items = []
        for report in dirs(DATA_DIR):
            runs_dir = DATA_DIR + "/" + report + "/Runs"
            for run in dirs(runs_dir):
                syms_dir = runs_dir + "/" + run + "/Snapshots"
                for sym in dirs(syms_dir):
                    snapshots_dir = syms_dir + "/" + sym +  "/__Snapshots__"
                    for component in dirs(snapshots_dir):
                        images_dir = snapshots_dir + "/" + component
                        for image in dirs(images_dir):
                            image_path = images_dir + "/" + image
                            items.append({
                                "Machine": report,
                                "Commit": run,
                                "Sym": sym,
                                "Component": component,
                                "ImageId": image,
                                "Snapshot": image_path
                            })
        df = pd.DataFrame(items)
        return df

    snaps = load_snapshots(DATA_DIR)
    return dirs, load_snapshots, snaps


@app.cell
def _(mo, snaps):
    # Данные

    def stat_info(name, title=None, caption=None):
        count = snaps[name].nunique()
        if title is None:
            title = name

        return mo.stat(label=title, value=count, caption=caption)

    mo.vstack([
        mo.md("## Собранные данные"),
        mo.hstack([
            stat_info("Machine", title="Тачек"),
            stat_info("Sym", title="Симуляторов", caption="*Только на одной тачке"),
            stat_info("Component", title="Компонентов"),
            stat_info("ImageId", title="Снапшотов в прогоне"),
            stat_info("Snapshot", title="Всего снапшотов")
        ]),
        mo.ui.table(data=snaps)
    ])
    return (stat_info,)


@app.cell
def _(snaps):
    # Селекторы данных

    def snaps_by(column):
        columns = ["Machine", "Sym", "Component", "ImageId", "Commit"]
        columns.remove(column)
        return snaps.pivot_table(index=column, columns=columns, values="Snapshot", aggfunc="first")
    return (snaps_by,)


@app.cell
def _(np, pd, pl):
    # Способы сравнивания снапшотов

    def compare_existence(image_path1, image_path2):
        is_image1 = not pd.isna(image_path1)
        is_image2 = not pd.isna(image_path2)

        if is_image1 == is_image2:
            return 0.0

        if is_image1:
            return 1.0
        else:
            return -1.0

    def compare_pixel_by_pixel_total(image_path1, image_path2):
        image1 = pl.Image.open(image_path1).convert("RGB")
        image2 = pl.Image.open(image_path2).convert("RGB")

        if image1.size != image2.size:
            return np.nan

        arr1 = np.array(image1)
        arr2 = np.array(image2)

        diff = np.sum(arr1 != arr2)
        return diff
    return compare_existence, compare_pixel_by_pixel_total


@app.cell
def _(itertools, pd, snaps_by):
    # Алгоритмы сравниваний

    def compare_adjacent_by(column, method=None):
        pivot = snaps_by(column)

        shifted = pivot.shift(-1).iloc[:-1]
        diff_pivot = pivot.iloc[:-1].combine(shifted, lambda s1, s2: s1.combine(s2, method))
        return diff_pivot.T

    def compare_first_by(column, method=None):
        pivot = snaps_by(column)
        first_row = pivot.iloc[0]
        diff_pivot = pivot.iloc[:-1].apply(lambda r: r.combine(first_row, method), axis=1)
        return diff_pivot.T

    def compare_combinations_by(column, method=None):
        pivot = snaps_by(column)
        pairs = list(itertools.combinations(pivot.index, 2))

        index = []
        rows = []
        for [a, b] in pairs:
            sa = pivot.loc[a]
            sb = pivot.loc[b]
            diff = sa.combine(sb, method)
            index.append([a, b])
            rows.append(diff)

        result = pd.DataFrame(rows, columns=pivot.columns, index=pd.MultiIndex.from_tuples(index))
        return result.T
    return compare_adjacent_by, compare_combinations_by, compare_first_by


@app.cell
def _(compare_adjacent_by, compare_pixel_by_pixel_total):
    diff_by_commits = compare_adjacent_by("Commit", method=compare_pixel_by_pixel_total)
    diff_by_commits
    return (diff_by_commits,)


@app.cell
def _(diff_by_commits):
    diff_by_commits.sum(axis=0)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
