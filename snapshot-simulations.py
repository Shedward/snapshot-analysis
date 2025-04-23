

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium", app_title="Snapshot тестирование")


@app.cell(hide_code=True)
def _():
    import os
    import re
    import tempfile
    import shutil
    import subprocess
    import json
    import pathlib
    import random
    import marimo as mo
    import pandas as pd
    import PIL as pl
    import plotly.express as px
    import numpy as np
    import itertools
    import xml.etree.ElementTree as ET
    return (
        ET,
        itertools,
        json,
        mo,
        np,
        os,
        pathlib,
        pd,
        pl,
        px,
        random,
        re,
        shutil,
        subprocess,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Snapshot тесты в ДС

        ## Зачем 
        - Верстку физически невозможно постоянно проверять вручную <span style="color: orange;">::ph:star-duotone::</span>
        - Автоматический регресс ДС <span style="color: orange;">::ph:star-half-duotone::</span>
        - Править верстку и рефакторить становится более безопасно
        - Когда заедет a11y, сможем похожим подходом защищать целостность текстовой-верстки
        """
    )
    return


@app.cell(hide_code=True)
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Открытые вопросы

        > https://miro.com/app/board/uXjVIV6oPjw=/

        - Насколько надежны/флакуют? <span style="color: red;">::ph:warning-duotone::</span>
        - Насколько долго гоняются?
        - Где хранить снапшоты?
        - Как их вообще писать?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Какой был план

        1. Выбрать N коммитов
        2. Выбрать несколько компонентов
        3. Написать для них снапшот тесты
        4. Прогнать тесты под N коммитов, на M машинах
        5. Собрать и посмотреть как снапшоты себя ведут
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Пишем тесты

        ## Выбор либы
        Выбранная либа - SwiftSnapshotTesting
        > [::ph:github-logo-duotone:: pointfree/swift-snapshot-testing](https://github.com/pointfreeco/swift-snapshot-testing)

        - От pointfree (как и Perception)
        - Жива и поддерживается
        - Поддерживает и UIKit и ViewController'ы и SwiftUI вьюхи. Можно добавить что-то свое
        - Можно менять практически все: от чего делать снимок, как делать снимок, как сравнивать

        В чистом виде тесты на ней выглядят вот так

        `swift
            func testWithSnapshotTesting() {
                let view = Button(label: "Label")
                    .style(.contrast)

                assertSnapshot(of: view, as: .image(/*доп настройки, если нужны*/))
            }
        `

        ### Доп. идеи <span style="color: blue;">::ph:sparkle-duotone::</span>

        - Есть плагины для a11y, можно попробовать их накатать/написать свой когда активно будем этим заниматься
        - Либа позволяет делать снапшоты любых данных которые можно сравнивать и сохранять. Можно применить к чему угодно, экраны, json'ы, сырые данные, и т.д.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Пишем тесты
        Были выбранны компоненты TextText, BuonButton, InputInput, ChipsChips и Se≤ctSelect.

        ### Тесты под компоненты-вьюхи (TextText, BuonButton, InputInput, ChipsChips)

        - Обычный UnitTest в модуле DesignSystem
        - Большое количество инвариантов поэтому пришлось накатать небольшую DSL'ку
            `swift
            SnapshotConfigurationSet()
                .all(\.property, in: ["id1": value1, "id2": value2]) // В полном виде
                .all(\.property2) // Для простых случаев когда по всем кейсам можно пройтись автоматически
                .allThemes() // Для популярных можно добавить кастомные
            `
        - Каждый параметр кратно умножает количество снапшотов, поэтому нужно быть аккуратнее, группировать параметры в отдельные тесты по смыслу (layout, colors, someSpecificCase)
        - id для значений нужен чтобы отличать снапшоты друг от друга

        `swift
            func testButtonLayout() {
                assertSnapshots(
                    of: Button(label: "Label", icon: Images.icon.bellFilled.size24),
                    configurationSet: SnapshotConfigurationSet()
                        .allLayouts([.sizeThatFits, .fixed(320, 64), .fixed(128, 64)])
                        .all(\.size, in: ["small": .small, "medium": .medium, "large": .large])
                        .all(\.isLoading)
                        .all(\.isStretched)
                )
            }

            func testButtonColors() {
                assertSnapshots(
                    of: Button(icon: Images.icon.bellFilled.size24),
                    configurationSet: SnapshotConfigurationSet()
                        .all(\.style, in: [
                            "neutral": .neutral,
                            "positive": .positive,
                            "negative": .negative,
                            "accent": .accent,
                            "contrast": .contrast
                        ])
                        .all(\.mode, in: ["primary": .primary, "secondary": .secondary, "tertiary": .tertiary])
                        .all(\.$isDisabled)
                )
            }
        `

        ### Результат
        <img src="public/snapshot_output.png">

        ### Нюансы <span style="color: red;">::ph:warning-duotone::</span>

        - swift-snapshot-testing содержит #canImport(Test∈g)#canImport(Testing) который true, из-за этого таргет начинает ожидать Testing.xcframework и падать с missingrequiredameworkmissing required framework. Причем в UnitTest'ах работает без этой проблемы, а в UITest'ах починить не смог. Пока как воркэрануд - форкнул и поубирал поддержку Swift Testing. Но возможно получится наколдовать что-то на уровне tuist'а
        - Нативный sizeThatFit работает странновато, некоторые компоненты нелогичного размера. Возможно баг в самих компонентах. Так же можно брать size из ManualComponent и переопределять
        - Сейчас компонент тестируется как SwiftUI вьюха с дефолтным контекстом, возможно правильнее будет тестировать компонент с полноценно настроенным контекстом

        ### Доп. идеи <span style="color: blue;">::ph:sparkle-duotone::</span>
        - Сверху, практически бесплатно можно навесить проверку что ManualComponent.size (почти)равен рассчитаному вручную
        - Так же, на уровне DSL'ки можно добавить любую проверку, которая автоматически раскатается на все описанные состояния компонентов
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Тесты под компоненты-экраны (Select)

        - UITest в UITests/Main
        - Работает через системный XCUIScreenshotProviding который просто делает скриншоты
        - Напрямую swift-snapshot-testing не поддерживает UITest'ы, но легко добавить парой экстеншнов
        `swift
        extension PageObject {
            @discardableResult
            public func assertSnapshot<Element: XCUIScreenshotProviding>(
                of keyPath: KeyPath<Self, Element>,
                ...
            ) { ... }
        }

        extension XCUIScreenshotProviding {
            @discardableResult
            public func assertSnapshot(
                _ name: String? = nil,
                ...
            ) { ... }
        }
        `
        - Количество снапшотов небольшое, но нужно их именовать и отличать
        `swift
        import SnapshotTesting
        import UITesting
        import XCTest

        final class SelectTestSuite: BaseTestCase {
            private lazy var deeplinkApp = serviceFactory.deeplinkAppService()

            override var shouldLaunchAppOnSetup: Bool {
                false
            }

            override func setUp() {
                super.setUp()
                continueAfterFailure = true
            }

            func test_Select() throws {
                application.launch()
                deeplinkApp.openDeeplinkDirectly(url: "hhios://magritte_showroom/select")

                application.windows.firstMatch.assertSnapshot("Select playground")
                application.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.23)).tap()

                guard application.staticTexts["Item 1"].waitForExistence(timeout: 5) else {
                    return
                }

                application.windows.firstMatch.assertSnapshot("Open select screen")
            }
        }
        `

        ### Результат
        <img src="public/screen_snapshot_output.png">

        ### Нюансы <span style="color: red;">::ph:warning-duotone::</span>
        - Хотелось бы открывать через диплинк
        - Основной источник флакования - статусбар, желательно его подрезать (или впринципе скриншотить только тестируемые области)
        - Второй источник флакования - анимации, их бы отключать

        ### Доп. идеи <span style="color: blue;">::ph:sparkle-duotone::</span>
        - При желании можно настраивать экраны через параметры диплинка, а не через плейграунд, будет проще и быстрее
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Гоняем тесты

        ## Как прогонялись тесты
        - Составлен список из 20 коммитов (PR to dev, с середины февраля по конец марта)
        - CI запускает скрипт на нескольких машинках
        - Результат скрипта - архив со снапшотами за эти 20 коммитов и технической информацией
        - Архивы со всех 5ти машин собираются и обрабатываются

        <img src="public/collection_process.png", width=60%>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Собранные данные""")
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

        pattern = r'^\s*([\w\s-]+):\s+(.+)$'
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
    return MACHINES, SNAPS, TEST_TIME, dirs


@app.cell(hide_code=True)
def _(MACHINES, mo):
    mo.vstack([
        mo.md("### Тачки"),
        mo.ui.table(data=MACHINES, freeze_columns_left=["Machine"])
    ])
    return


@app.cell(hide_code=True)
def _(SNAPS, mo):
    # Данные

    def stat_info(name, title=None, caption=None, snaps=SNAPS):
        count = snaps[name].nunique()
        if title is None:
            title = name

        return mo.stat(label=title, value=count, caption=caption)

    mo.vstack([
        mo.md("### Снапшоты"),
        mo.hstack([
            stat_info("Machine", title="Тачек"),
            stat_info("Sym", title="Симуляторов"),
            stat_info("Component", title="Компонентов"),
            stat_info("ImageId", title="Снапшотов в прогоне"),
            stat_info("Snapshot", title="Всего снапшотов")
        ]),
        mo.ui.table(data=SNAPS),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Обрабатываем данные""")
    return


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
    return compare_pixel_by_pixel_diff, compare_pixel_by_pixel_total


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
    return compare_adjacent_by, compare_combinations_by


@app.cell
def _(compare_adjacent_by, compare_pixel_by_pixel_total):
    diff_by_commits = compare_adjacent_by("Commit", method=compare_pixel_by_pixel_total)
    return (diff_by_commits,)


@app.cell(hide_code=True)
def _(diff_by_commits, mo):
    mo.vstack([
        mo.md("""
        ## Разница по коммитам
        Затем чтобы проверить флакование - сравниваем как снапшоты изменяются во времени.

        В таблице - суммарное количество отличающихся пикселей (со всех 5 машин!). Метод сравнивания пикселей тут без допуска - должно совпадать 1 к 1.
        """),
        mo.ui.table(data=diff_by_commits.groupby("Component").sum()),
        mo.md("""
        Видим что в целом постоянного флакования нет. Есть точечные различия у некоторых компонентов.
        Чтобы посмотреть подробнее - визуализируем
        """)
    ])
    return


@app.cell(hide_code=True)
def _(compare_adjacent_by, compare_pixel_by_pixel_diff):
    diff_images_by_commits = compare_adjacent_by(
        "Commit", 
        method=compare_pixel_by_pixel_diff
    )
    return (diff_images_by_commits,)


@app.cell(hide_code=True)
def _(diff_images_by_commits, mo, pd, pl):

    def compose_images_vertically(images):
        images = [i for i in images if pd.notnull(i)][:3]
        if len(images) == 0:
            return None

        # Ensure all images are the same width (optional: resize or pad if needed)
        widths, heights = zip(*(img.size for img in images))
        total_height = sum(heights)
        max_width = max(widths)

        composite = pl.Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for img in images:
            composite.paste(img, (0, y_offset))
            y_offset += img.height

        return composite

    mo.vstack([
        mo.md("### Diff снапшотов"),
        mo.ui.table(data=diff_images_by_commits.groupby("Component").agg(compose_images_vertically))
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Разница между тачками

        Еще одним вопросом было - будут ли флаковать снапшоты при прогоне на различных тачках.
        Чтобы проверить - сравним все снапшоты каждое с каждым
        """
    )
    return


@app.cell
def _(compare_combinations_by, compare_pixel_by_pixel_total):
    diff_by_machines = compare_combinations_by("Machine", method=compare_pixel_by_pixel_total)
    return (diff_by_machines,)


@app.cell
def _(diff_by_machines):
    diff_by_machines.sum(axis=0).unstack(level=0)
    return


@app.cell(hide_code=True)
def _(SNAPS, compare_combinations_by, compare_pixel_by_pixel_diff):
    compare_combinations_by(
        "Machine", 
        method=compare_pixel_by_pixel_diff, 
        snaps=SNAPS[SNAPS["Component"] == "SelectTestSuit"]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Как долго гоняются

        Для того чтобы понять насколько снапшоты утяжелят UI я собрал время прогона тестов с junit репортов
        """
    )
    return


@app.cell(hide_code=True)
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
        mo.ui.table(data=pd.DataFrame({"Mean": test_time.mean(), "Rel. Std. Dev., %": test_time.std() / test_time.mean() * 100.0}))
    ])
    return


@app.cell
def _():
    def mult_components_metrics(easy=0, hard=0):
        easy * 47 + hard * 11 # 47 - вьюх, 11 - экранов

    mult_components_metrics(20, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Данные в целом получились довольно мусорные.
        Но можно оценить порядки. В целом UnitTests'ы гоняются за секунды, а UI тесты - за десятки секунд.
        Если грубо помножить величины на типы компонентов, можно предположить 

        <img src="public/component_type.png" width=50%/>

        что сами тесты ДС уйдет порядка **5-10 минут**.

        Но, помимо тестов - **значительную часть прогона будет составлять подготовка к сборке, сборка, подготова симуляторов** и т.д.
        Когда я гонял скрипт среднее время 20 прогонов было порядка - 1.5 - 2 часа.
        Т.е. **на CI один прогон (без чекаута, но с бутсрапом) - около 6-10 минут**

        Еще если локально один прогон (без пересборки) занимает порядка 30 секунд.


        Не знаю как эти цифры сопоставить без полноценно работающего пайплайна (очень много зависит от него)

        Но могу предположить что **на один прогон всех снапшотов потребуется порядка десятка минут**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Где запускать?
        В рамках эксперемента - я гонял эти тесты отдельным планом.
        Но в теории они достаточно хорошо вписываются в стандартные UnitTest и UITest планы.
        При желании можно гонять прям в них.
        Либо отделить как сейчас.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Где хранить артефакты?

        Средний размер снапшота
        """
    )
    return


@app.cell(hide_code=True)
def _(SNAPS, os):
    def file_sizes(snaps=SNAPS):
        output = snaps.copy()
        output['Size'] = output['Snapshot'].apply(lambda p: os.path.getsize(p) if os.path.exists(p) else None)
        return output

    def format_size(bytes):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(bytes) < 1024:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024

    SNAPS_SIZE=file_sizes()
    SNAPS_SIZE.groupby("Component")["Size"].mean().apply(format_size)
    return SNAPS_SIZE, format_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Сумарный размер снапшотов для компонента""")
    return


@app.cell(hide_code=True)
def _(SNAPS_SIZE, format_size, pd):
    def size_by_component(snap_size=SNAPS_SIZE):
        by_components = snap_size.groupby(['Component', 'Commit', 'Machine', 'Sym'])['Size'].sum().groupby("Component")
        return pd.DataFrame({"Mean": by_components.mean().apply(format_size), "Rel.Std.Dev., %": by_components.std() / by_components.mean() * 100.0})

    size_by_component()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Общий вес снапшотов на все компоненты""")
    return


@app.cell(hide_code=True)
def _(SNAPS_SIZE, format_size, pd):
    def total_size(snap_size=SNAPS_SIZE):
        by_components = snap_size.groupby(['Commit', 'Machine', 'Sym'])['Size'].sum()
        return pd.DataFrame({"Mean": format_size(by_components.mean()), "Rel.Std.Dev., %": by_components.std() / by_components.mean() * 100.0}, index=["Total"])

    total_size()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Влияние на Git

        // TODO
        """
    )
    return


@app.cell
def _(SNAPS, json, metrics, os, pathlib, pd, random, shutil, subprocess):
    def run(*cmd, at=None, expected_statuses=[], check=True):
        print("     > " + " ".join(cmd))
        output = subprocess.run(cmd, cwd=at, capture_output=True, text=True, check=check)
        if output.returncode != 0:
            print(f"    ! Failed status {output.returncode}:\n {output.stderr}")
        return output

    def collect_metrics(dir):
        def get_size(path):
            return sum(p.stat().st_size for p in pathlib.Path(path).rglob('*'))

        def dir_size(dir):
            return { "dir_size": get_size(dir) }

        def git_dir_size(dir):
            return { "git_dir_size": get_size(dir + "/" + ".git") }

        def git_sizer(dir):
            output = run("git-sizer", "--json", "--no-progress", at=dir)
            data = json.loads(output.stdout)
            return data

        def git_counters(dir):
            output = run("git", "count-objects", "-v", at=dir)
            lines = output.stdout.strip().split("\n")
            stats = {}

            for line in lines:
                key, value = line.split(":")
                stats["git_counter_" + key.strip()] = value.strip()

            return stats

        metrics = [
            dir_size,
            git_dir_size,
            git_counters,
            git_sizer
        ]

        metrics_result = {}
        for metric in metrics:
            metrics_result = metrics_result | metric(dir)

        return metrics_result


    def run_git_snapshot_simulation(
        snaps=SNAPS,
        working_dir="/tmp/SnapshotExperiments",
        output="Data/git_simulations.csv",
        runs_count = 30,
        snapshots_count = 200,
        commits_count = 30,
        exps=[
            { 
                "name": "OriginalRepository", 
                "with_snapshots": False, 
                "with_commits": True 
            },
            {
                "name": "SnapshotsInGit", 
                "with_snapshots": True, 
                "with_commits": True 
            },
            { 
                "name": "SnapshotsInLFS",
                "init_lfs": True,
                "with_snapshots": True,
                "with_commits": True 
            },
            { 
                "name": "OnlySnapshotsInGit", 
                "with_snapshots": True,
                "with_commits": False 
            },
        ]
    ):
        """
        Для подготовки

            export BEGIN_COMMIT=49ffd59953d
            export END_COMIMIT=f781db47392

            mkdir -p "tmp/SnapshotExperiments"
            cd "tmp/SnapshotExperiments"

            # Подготовка Source
            git clone git@forgejo.pyn.ru:hhru/ios-apps.git Source
            (cd Source && git branch snapshot-tests/begin $BEGIN_COMMIT)
            (cd Source && git branch snapshot-tests/end $END_COMMIT)
            (cd Source && git rev-list --reverse snapshot-tests/end ^snapshot-tests/begin) > commits.txt


            # Подготовка WithoutSnapshots (полная копия репозитория)
            git clone file://$(realpath Source) --branch snapshot-tests/begin WithoutSnapshots

            # Подготовка WithGit (shallow copy от snapshot-tests/begin)
            git clone file://$(realpath Source) --branch snapshot-tests/begin WithGit

            # Подготовка WithLFS (shallow copy от snapshot-tests/begin c LFS)
            git clone file://$(realpath Source) WithLFS --branch snapshot-tests/begin --depth 1
            mkdir -p WithLFS.lfs

            (cd WithLFS && git lfs install)
            (cd WithLFS && git lfs track "__Snapshots__/**")
            (cd WithLFS && git config --add lfs.customtransfer.lfs-folder.path lfs-folderstore)
            (cd WithLFS && git config --add lfs.standalonetransferagent lfs-folder)
            (cd WithLFS && git config --add lfs.customtransfer.lfs-folder.args $(realpath ../WithLFS.lfs))


            # git-force-cherry-pick
            git cherry-pick $1 --no-commit || (
              echo "Conflict — auto-resolving with cherry-pick's changes (theirs)"
              for file in $(git diff --name-only --diff-filter=U); do
                if git show ":3:$file" >/dev/null 2>&1; then
                  git checkout --theirs -- "$file"
                  git add "$file"
                else
                  git rm --cached --quiet --ignore-unmatch "$file"
                  rm -f "$file"
                fi
              done
            )
        """

        def prepare_exps_dir():
            print("  -- Prepearing exps dir")
            exps_dir = working_dir + "/Exps"
            if os.path.exists(exps_dir):
                shutil.rmtree(exps_dir)
            os.makedirs(exps_dir, exist_ok=True)
            return exps_dir

        def init_repo(exp):
            name = exp["name"]
            print(f"  -- Init repo for exp {name}")

            exps_dir = working_dir + "/Exps"
            exp_dir = exps_dir + "/" + exp["name"]
            source_git_url = "file://" + os.path.abspath(working_dir + "/Source")
            run("git", "clone", source_git_url, "--branch", "snapshot-tests/begin", exp["name"], at=exps_dir)

            if exp.get("init_lfs"):
                run("git", "lfs", "install", at=exp_dir)
                run("git", "lfs", "track", "__Snapshots__/**", at=exp_dir)

                lfs_path = os.path.abspath(exp_dir + ".lfs")
                os.makedirs(lfs_path, exist_ok=True)

                run("git", "config", "--add", "lfs.customtransfer.lfs-folder.path", "lfs-folderstore", at=exp_dir)
                run("git", "config", "--add", "lfs.standalonetransferagent", "lfs-folder", at=exp_dir)
                run("git", "config", "--add", "lfs.customtransfer.lfs-folder.args", lfs_path, at=exp_dir)

        def add_salt(dir, salt):
            print(" -- Add salt")
            with open(dir + "/salt.txt", "a") as f:
                f.write(salt + "\n")

        def generate_snapshots(to_dir, snapshots):
            print(f"  -- Copy snapshots {len(snapshots)}")
            os.makedirs(to_dir, exist_ok=True)

            def mutate_binary(file_path):
                with open(file_path, 'rb') as f:
                    data = bytearray(f.read())

                index = random.randint(8, len(data) - 1)

                bit_to_flip = 1 << random.randint(0, 7)
                data[index] ^= bit_to_flip

                with open(file_path, 'wb') as f:
                    f.write(data)

            for snapshot in snapshots:
                shutil.copy2(snapshot, to_dir)
                mutate_binary(to_dir + "/" + os.path.basename(snapshot))



        def cherry_pick(dir, commit):
            run("git", "force-cherry-pick", commit, check=False, at=dir)
            run("git", "checkout", "--theirs",  ".", at=dir)
            add_salt(dir, commit)
            run("git", "add", ".", at=dir)
            run("git", "commit", "-m", commit, at=dir)

        original_commits = []
        with open(working_dir + "/commits.txt", "r") as f:
            original_commits = [line.strip() for line in f]

        print(f"Prepearing")
        print("---")
        print(f"  -- Found {len(original_commits)} original commits")

        exps_dir = prepare_exps_dir()
        for exp in exps:
            init_repo(exp)

        with open(working_dir + "/output.json", "w") as out:
            comma = ""
            out.write("[\n")
            for run_index in range(runs_count):
                print()
                print(f"Run {run_index}")
                print("---")

                snapshots = snaps["Snapshot"].sample(n=snapshots_count).tolist()
                commits_to_add = original_commits[:commits_count]
                original_commits = original_commits[commits_count:]

                for exp in exps:
                    exp_name = exp["name"]
                    exp_dir = exps_dir + "/" + exp_name

                    print(f" | Exp {exp_name}")

                    metrics_result = {"Num": run_index, "Exp": exp_name } | collect_metrics(metrics)

                    json_output = comma + json.dumps(metrics_result) + "\n"
                    out.write(json_output)

                    if exp["with_commits"]:
                        for commit in commits_to_add:
                            print(f" -- Cherry pick {commit}")
                            cherry_pick(exp_dir, commit)

                    if exp["with_snapshots"]:
                        generate_snapshots(exp_dir + "/__Snapshots__", snapshots)
                        run("git", "add", ".", at=exp_dir)
                        run("git", "commit", "-m", "Snapshots", at=exp_dir, check=False)

                    comma = ","

            out.write("]\n")

        print("===")
        print(f"Finished!")
        print("---")



    # run_git_snapshot_simulation()

    def size_test_metrics():
        origin_size_metrics = {"Num": 0, "Exp": "SizeTestOrigin"} | collect_metrics("/tmp/SnapshotExperiments/SizeTestOrigin")
        year_ago_size_metrics = {"Num": 0, "Exp": "SizeTestYearAgo"} | collect_metrics("/tmp/SnapshotExperiments/SizeTestYearAgo")
        return pd.DataFrame([origin_size_metrics, year_ago_size_metrics])
    return (size_test_metrics,)


@app.cell
def _(pd, size_test_metrics):
    GIT_EXPERIMENTS_METRICS = pd.read_json("git_metrics_output.json")
    GIT_SIZE_TEST_METRICS = size_test_metrics()

    GIT_METRICS = pd.concat([GIT_EXPERIMENTS_METRICS, GIT_SIZE_TEST_METRICS])
    GIT_METRICS
    return (GIT_METRICS,)


@app.cell
def _(GIT_METRICS, format_size, pd):
    metric_candidates = [col for col in GIT_METRICS.columns if col not in ['Num', 'Exp']]
    GIT_NUM_METRICS = [
        col for col in metric_candidates
        if pd.api.types.is_numeric_dtype(GIT_METRICS[col].iloc[0])
    ]

    GIT_SIZE_METRICS = [
        col for col in GIT_NUM_METRICS
        if "size" in col
    ]

    def format_size_metrics(df=GIT_METRICS):
        df = df.copy()
        df[GIT_SIZE_METRICS] = df[GIT_SIZE_METRICS].map(format_size)
        return df
    return GIT_NUM_METRICS, format_size_metrics


@app.cell
def _(format_size_metrics):
    format_size_metrics()
    return


@app.cell
def _(GIT_NUM_METRICS, mo):
    metric_name_dropdown = mo.ui.dropdown(options=GIT_NUM_METRICS, label="Git property")
    return (metric_name_dropdown,)


@app.cell
def _(GIT_METRICS, metric_name_dropdown, mo, px):
    mo.vstack([
        mo.md("## Метрики со всех экспериментов"),
        metric_name_dropdown,
        mo.ui.plotly(px.line(GIT_METRICS, x="Num", y=metric_name_dropdown.value, color="Exp", markers=True, range_y=[0, None]))
    ])
    return


@app.cell
def _(GIT_METRICS, GIT_NUM_METRICS, format_size_metrics, pd):
    def git_compare(exp1, exp2, metrics = GIT_NUM_METRICS, cmp = lambda l, r: l - r):
        p = GIT_METRICS.pivot(index="Num", columns="Exp", values=metrics)
        diffs = {}
        for metric in metrics:
            diffs[metric] = cmp(p[(metric, exp1)], p[(metric, exp2)])
        return pd.DataFrame(diffs)

    size_report = format_size_metrics(git_compare("SnapshotsInGit","OriginalRepository"))
    size_report.iloc[-1]
    return (git_compare,)


@app.cell
def _(git_compare):
    rel_comp = git_compare("SnapshotsInGit","OriginalRepository", cmp=lambda l, r: (l - r) / l * 100)
    rel_comp.iloc[-1]
    return


@app.cell
def _(format_size_metrics, git_compare):
    format_size_metrics(git_compare("SizeTestOrigin","SizeTestYearAgo"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Выводы

        - Снапшоты на ДС работать будут
        - Будут работать и на вьюхи и на экраны
        - В нормальных условиях снапшоты не флакуют, если
            - Не гонять на Intel тачках
            - Обрезать статусбар
            - Отключить анимации 
        - Прогон снапшотов по времени должен занимать порядка десятка минут
        - Снапшоты можно хранить в гите

        Итог: **Снапшоты в ДС завозим**
        """
    )
    return


if __name__ == "__main__":
    app.run()
