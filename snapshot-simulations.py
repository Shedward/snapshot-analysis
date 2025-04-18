import marimo

__generated_with = "0.11.31"
app = marimo.App(
    width="medium",
    app_title="Snapshot тестирование",
    layout_file="layouts/snapshot-simulations.slides.json",
)


@app.cell
def _():
    import os
    import re
    import marimo as mo
    import pandas as pd
    import PIL as pl
    import numpy as np
    import itertools
    import xml.etree.ElementTree as ET
    return ET, itertools, mo, np, os, pd, pl, re


@app.cell
def _(mo):
    mo.md("""# Snapshot тесты""")
    return


@app.cell
def _():
    DATA_DIR = "Data"
    return (DATA_DIR,)


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Собираем данные 

        - Были выбраны 20 коммитов в периоде с начала февраля по конец марта
        - Были выбраны несколько компонентов: `Text`, `Button`, `Input`, `Chips` и `Select`
        - Для них были написаны тесты - `Text`, `Button`, `Input`, `Chips` - через UnitTest'ы, `Select` - через UITest
        - Скриптом был прогнаны тесты для всех 20 коммитов вподряд
        - Планом на Bamboo скрипт был прогнан на 5ти машинах на CI
        """
    )
    return


@app.cell(hide_code=True)
def _(DATA_DIR, ET, os, pd, re):
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

    def extract_test_times(path):
        tree = ET.parse(path + "/TestReports/UITestsJUnitReport.xml" )
        root = tree.getroot()

        test_times = {}

        for testcase in root.iter('testcase'):
            name = testcase.attrib.get('classname') or testcase.attrib.get('name')
            time = float(testcase.attrib.get('time', 0))
            if name:
                test_times[name] = time

        return test_times
    

    def load_snapshots(data_dir=DATA_DIR):
        items = []
        machines = []
        test_time = []
        for machine in dirs(data_dir):
            machine_dir = data_dir + "/" + machine
            runs_dir = machine_dir + "/Report/Runs"
            machines.append(extract_machine_info(machine_dir) | {"Machine": machine})
            for run in dirs(runs_dir):
                syms_dir = runs_dir + "/" + run + "/Snapshots"
                for sym in dirs(syms_dir):
                    sym_dir = syms_dir + "/" + sym
                    snapshots_dir = sym_dir +  "/__Snapshots__"
                    test_time.append(extract_test_times(sym_dir) | {"Machine": machine, "Commit": run, "Sym": sym})
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
        test_time = pd.DataFrame(test_time)
        return (snaps, machines, test_time)

    (SNAPS, MACHINES, TEST_TIME) = load_snapshots()
    return (
        MACHINES,
        SNAPS,
        TEST_TIME,
        dirs,
        extract_key_values,
        extract_machine_info,
        extract_test_times,
        load_snapshots,
    )


@app.cell(hide_code=True)
def _(SNAPS, mo):
    # Данные

    def stat_info(name, title=None, caption=None, snaps=SNAPS):
        count = snaps[name].nunique()
        if title is None:
            title = name

        return mo.stat(label=title, value=count, caption=caption)

    mo.vstack([
        mo.md("## Собранные данные"),
        mo.md("### Снапшоты"),
        mo.hstack([
            stat_info("Machine", title="Тачек"),
            stat_info("Sym", title="Симуляторов", caption="*Только на одной тачке"),
            stat_info("Component", title="Компонентов"),
            stat_info("ImageId", title="Снапшотов в прогоне"),
            stat_info("Snapshot", title="Всего снапшотов")
        ]),
        mo.ui.table(data=SNAPS),
    ])
    return (stat_info,)


@app.cell(hide_code=True)
def _(MACHINES, mo):
    mo.vstack([
        mo.md("## Собранные данные"),
        mo.md("### Тачки"),
        mo.ui.table(data=MACHINES, freeze_columns_left=["Machine"])
    ])
    return


@app.cell
def _(TEST_TIME, mo):
    mo.vstack([
        mo.md("## Собранные данные"),
        mo.md("### Время отдельных тестов"),
        mo.ui.table(data=TEST_TIME)
    ])
    return


@app.cell
def _(TEST_TIME, mo, pd):
    def clean_tests_time(df, prefix="ApplicantHHUITests"):
        for col in df.columns:
            if col.startswith(f"{prefix}."):
                base_col = col[len(prefix)+1:]
                df[base_col] = df.get(base_col).combine_first(df[col]) if base_col in df.columns else df[col]
                df.drop(columns=[col], inplace=True)
        all_components = ["ButtonSnapshotTests", "ChipsSnapshotTests", "InputSnapshotTests", "TextSnapshotTests", "SelectTestSuite"]
        df = df[all_components + ["Machine", "Commit", "Sym"]]
        df = df.set_index(["Machine", "Commit", "Sym"])
        return df

    test_time = clean_tests_time(TEST_TIME)



    mo.vstack([
        mo.md("### Отчищенное время прогонов"),
        mo.ui.table(data=test_time),

        mo.md("### Время по компонентам (но по факту это без сравнения)"),
        mo.ui.table(data=pd.DataFrame({"Mean": test_time.mean(), "Rel. Std": test_time.std() / test_time.mean() * 100.0}))
    ])
    return clean_tests_time, test_time


@app.cell(hide_code=True)
def _(SNAPS):
    # Селекторы данных

    def snaps_by(column, snaps=SNAPS):
        columns = ["Machine", "Sym", "Component", "ImageId", "Commit"]
        columns.remove(column)
        return snaps.pivot_table(index=column, columns=columns, values="Snapshot", aggfunc="first")
    return (snaps_by,)


@app.cell(hide_code=True)
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
            max_width = max(image1.width, image2.width)
            max_height = max(image1.height, image2.height)
    
            def pad_image(image, target_width, target_height):
                padded = pl.Image.new("RGB", (target_width, target_height), (255, 0, 255))
                padded.paste(image, (0, 0))
                return padded
    
            image1_padded = pad_image(image1, max_width, max_height)
            image2_padded = pad_image(image2, max_width, max_height)
            image1 = image1_padded
            image2 = image2_padded

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


@app.cell(hide_code=True)
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
    return (diff_by_commits,)


@app.cell(hide_code=True)
def _(diff_by_commits, mo):
    mo.vstack([
        mo.md("## Разница по коммитам"),
        mo.ui.table(data=diff_by_commits.groupby("Component").sum())
    ])
    return


@app.cell
def _(compare_adjacent_by, compare_pixel_by_pixel_diff):
    diff_images_by_commits = compare_adjacent_by(
        "Commit", 
        method=compare_pixel_by_pixel_diff
    )
    return (diff_images_by_commits,)


@app.cell(hide_code=True)
def _(diff_images_by_commits, mo):
    mo.vstack([
        mo.md("## Разница по коммитам"),
        mo.md("В картинках"),
        mo.ui.table(data=diff_images_by_commits)
    ])
    return


@app.cell(hide_code=True)
def _(compare_combinations_by, compare_pixel_by_pixel_total):
    diff_by_machines = compare_combinations_by("Machine", method=compare_pixel_by_pixel_total)
    return (diff_by_machines,)


@app.cell(hide_code=True)
def _(diff_by_machines, mo, pd):
    def adjacent_comparation(diff):
        diff = diff.sum(axis=0)
        index = diff.index.str.split(' x ').map(tuple)
        index = pd.MultiIndex.from_tuples(index, names=["from", "to"])
        diff.index = index
        return diff.unstack(level='from')

    mo.vstack([
        mo.md("## Отличие между тачками"),
        mo.ui.table(data=adjacent_comparation(diff_by_machines))
    ])
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
