import matplotlib.pyplot as plt
import datetime
import json
import pandas as pd
import altair as alt
from altair_saver import save

TITLE_DATE = "%d-%m-%y %H:%M:%S"
FILE_DATE = "%d-%m-%y_%H:%M:%S"
date = datetime.datetime.now().strftime

title = lambda hist: f"Training session {date(TITLE_DATE)} - {hist.epoch[-1] + 1} epochs"


def plot_history(history, filename=None, backend="matplotlib"):
    """
    Dispatch to a specific history plotter based on chosen backends
    Backend options are matplotlib and altair
    """

    if backend == "altair":
        _plot_history_altair(history, filename)
        return
    elif backend == "matplotlib":
        _plot_history_matplotlib(history, filename)
        return
    else:
        raise NotImplementedError("Available backends are matplotlib and altair")


def _plot_history_matplotlib(history, filename=None):

    if filename is None:
        filename = date(FILE_DATE)

    for key, value in history.history.items():

        plt.plot(value, label=key)

    plt.title(title(history))
    plt.legend()
    plt.savefig(filename)


def _plot_history_altair(history, filename=None):

    if filename is None:
        filename = date(FILE_DATE) + ".png"

    data = history.history
    data["epoch"] = history.epoch
    data = pd.DataFrame(data).melt("epoch")
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(x=alt.X('epoch', title='Epoch'), y=alt.Y('value', title='Value'), color="variable")
        .properties(title=title(history))
        .configure_legend(
            title=None
        )
    )
    with open(filename, "wb") as f:
        save(chart, f)


def dump_history(history, filename=None):

    date_ = datetime.datetime.now().strftime(date(FILE_DATE))

    history.model = None

    if filename is None:
        filename = date_ + ".json"

    with open(filename, "w") as f:
        json.dump(history.__dict__, f)
