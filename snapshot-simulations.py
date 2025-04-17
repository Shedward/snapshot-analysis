import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium", app_title="Snapshot тестирование")


@app.cell
def _():
    import os
    import re
    import marimo as mo
    import pandas as pd
    import PIL as pl
    import numpy as np
    import itertools
    return itertools, mo, np, os, pd, pl, re


@app.cell
def _(mo):
    mo.md("""# Snapshot тесты""")
    return


@app.cell
def _():
    DATA_DIR = "Data"
    return (DATA_DIR,)


@app.cell
def _(DATA_DIR, dirs, os, pl):
    # Методы обработки данных

    def copy_component(component_name, new_component_name, transforms=[]):
        items = []
        for report in dirs(DATA_DIR):
            runs_dir = DATA_DIR + "/" + report + "/Report/Runs"
            for run in dirs(runs_dir):
                syms_dir = runs_dir + "/" + run + "/Snapshots"
                for sym in dirs(syms_dir):
                    snapshots_dir = syms_dir + "/" + sym +  "/__Snapshots__"
                    source_dir = snapshots_dir + "/" + component_name
                    dest_dir = snapshots_dir + "/" + new_component_name 

                    os.makedirs(dest_dir, exist_ok=True)
                    for filename in os.listdir(source_dir):
                        if filename.lower().endswith(".png"):
                            src_image = source_dir + "/" + filename
                            dest_image = dest_dir + "/" + filename

                            with pl.Image.open(src_image) as img:
                                for transform in transforms:
                                    img = transform(img)
                                img.save(dest_image)

    def cover_statusbar(image):
        from PIL import ImageDraw

        image = image.copy()
        draw = pl.ImageDraw.Draw(image)
        draw.rectangle([0, 0, image.width, 90], fill=(255, 0, 255, 255))
        return image

    def crop_bottom_1000(image):
        width, height = image.size
        top = max(0, height - 1000)  # Avoid negative crop if image is shorter
        return image.crop((0, top, width, height))

    # copy_component("SelectTestSuit", "SelectTestSuitWithoutStatusBar", transforms=[cover_statusbar])
    # copy_component("SelectTestSuit", "SelectTestSuitOnlyBottomSheet", transforms=[crop_bottom_1000])
    # copy_component("SelectTestSuit", "SelectTestSuit", transforms=[crop_bottom_1000])
    return copy_component, cover_statusbar, crop_bottom_1000


@app.cell
def _(DATA_DIR, os, pd, re):
    # Загружаем собранные снапшоты

    def dirs(path):
        dirs = os.listdir(path)
        return [d for d in dirs if not d.startswith(".")]

    def extract_key_values(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
    
        pattern = r'^\s*([\w\s\(\)-]+):\s+(.+)$'
        matches = re.findall(pattern, text, re.MULTILINE)
        return dict(matches)

    def extract_machine_info(path):
        hardware = extract_key_values(path + "/Report/hardware.txt")
        software = extract_key_values(path + "/Report/sw_vers.txt")
        return hardware | software
    

    def load_snapshots(data_dir=DATA_DIR):
        items = []
        machines = []
        for machine in dirs(data_dir):
            runs_dir = data_dir + "/" + machine + "/Report/Runs"
            machines.append(extract_machine_info(data_dir + "/" + machine) | {"Machine": machine})
            for run in dirs(runs_dir):
                syms_dir = runs_dir + "/" + run + "/Snapshots"
                for sym in dirs(syms_dir):
                    snapshots_dir = syms_dir + "/" + sym +  "/__Snapshots__"
                    for component in dirs(snapshots_dir):
                        images_dir = snapshots_dir + "/" + component
                        for image in dirs(images_dir):
                            image_path = images_dir + "/" + image
                            items.append({
                                "Machine": machine,
                                "Commit": run,
                                "Sym": sym,
                                "Component": component,
                                "ImageId": image,
                                "Snapshot": image_path
                            })
        snaps = pd.DataFrame(items)
        machines = pd.DataFrame(machines)
        return (snaps, machines)

    (SNAPS, MACHINES) = load_snapshots()
    return (
        MACHINES,
        SNAPS,
        dirs,
        extract_key_values,
        extract_machine_info,
        load_snapshots,
    )


@app.cell
def _(MACHINES):
    MACHINES
    return


@app.cell
def _(SNAPS, mo):
    # Данные

    def stat_info(name, title=None, caption=None, snaps=SNAPS):
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
        mo.ui.table(data=SNAPS)
    ])
    return (stat_info,)


@app.cell
def _(SNAPS):
    # Селекторы данных

    def snaps_by(column, snaps=SNAPS):
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

    def compare_pixel_by_pixel_mask(image_path1, image_path2):
        image1 = pl.Image.open(image_path1).convert("RGB")
        image2 = pl.Image.open(image_path2).convert("RGB")

        if image1.size != image2.size:
            return np.nan

        arr1 = np.array(image1)
        arr2 = np.array(image2)

        diff_mask = np.any(arr1 != arr2, axis=-1).astype(np.uint8) * 255
        return pl.Image.fromarray(diff_mask, mode="L")

    def compare_pixel_by_pixel_diff(image_path1, image_path2):
        image1 = pl.Image.open(image_path1).convert("RGB")
        image2 = pl.Image.open(image_path2).convert("RGB")

        if image1.size != image2.size:
            raise ValueError("Images must be the same size to compare")

        arr1 = np.asarray(image1)
        arr2 = np.asarray(image2)

        if np.array_equal(arr1, arr2):
            return None

        diff_mask = np.any(arr1 != arr2, axis=-1)

        mask_array = (diff_mask * 255).astype(np.uint8)
        mask_rgb = np.stack([mask_array]*3, axis=-1)  # Grayscale mask in RGB
        mask_image = pl.Image.fromarray(mask_rgb, mode="RGB")

        width, height = image1.size
        collage = pl.Image.new("RGB", (width * 3, height))
        collage.paste(image1, (0, 0))
        collage.paste(mask_image, (width, 0))
        collage.paste(image2, (width * 2, 0))

        return collage
    return (
        compare_existence,
        compare_pixel_by_pixel_diff,
        compare_pixel_by_pixel_mask,
        compare_pixel_by_pixel_total,
    )


@app.cell
def _(SNAPS, itertools, pd, snaps_by):
    # Алгоритмы сравниваний

    def compare_adjacent_by(column, method=None, snaps=SNAPS):
        pivot = snaps_by(column, snaps=snaps)

        shifted = pivot.shift(-1).iloc[:-1]
        diff_pivot = pivot.iloc[:-1].combine(shifted, lambda s1, s2: s1.combine(s2, method))
        return diff_pivot.T

    def compare_first_by(column, method=None, snaps=SNAPS):
        pivot = snaps_by(column, snaps=snaps)
        first_row = pivot.iloc[0]
        diff_pivot = pivot.iloc[:-1].apply(lambda r: r.combine(first_row, method), axis=1)
        return diff_pivot.T

    def compare_combinations_by(column, method=None, snaps=SNAPS):
        pivot = snaps_by(column, snaps=snaps)
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
    diff_by_commits.groupby("Component").sum()
    return


@app.cell
def _(compare_combinations_by, compare_pixel_by_pixel_total):
    diff_by_machines = compare_combinations_by("Machine", method=compare_pixel_by_pixel_total)
    diff_by_machines
    return (diff_by_machines,)


@app.cell
def _(diff_by_machines, pd):
    def adjacent_comparation(diff):
        diff = diff.sum(axis=0)
        index = diff.index.str.split(' x ').map(tuple)
        index = pd.MultiIndex.from_tuples(index, names=["from", "to"])
        diff.index = index
        return diff.unstack(level='from')

    adjacent_comparation(diff_by_machines)
    return (adjacent_comparation,)


@app.cell
def _(SNAPS, compare_combinations_by, compare_pixel_by_pixel_diff):
    compare_combinations_by(
        "Machine", 
        method=compare_pixel_by_pixel_diff, 
        snaps=SNAPS[SNAPS["Component"] == "SelectTestSuit"]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
