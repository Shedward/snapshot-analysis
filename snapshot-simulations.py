import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium", app_title="Snapshot тестирование")


@app.cell
def _():
    import marimo as mo
    import pandas
    return mo, pandas


@app.cell
def _(mo):
    mo.md("""# Snapshot тестирование""")
    return


@app.cell
def _():
    DATA_DIR = "Data/"
    return (DATA_DIR,)


@app.cell
def _():
    class RunLoader:
        def print(self):
            print("Hello")
    return (RunLoader,)


@app.cell
def _(RunLoader):
    RunLoader().print()
    return


if __name__ == "__main__":
    app.run()
