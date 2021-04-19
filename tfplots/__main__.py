import matplotlib.pyplot as plt
import datetime
import json
import pandas as pd
import altair as alt
from altair_saver import save


def plot_history(history, filename=None, backend='matplotlib'):

    if backend == 'altair':
        _plot_history_altair(history, filename)
        return 

    date = datetime.datetime.now().strftime("%d-%m-%y-%H:%M:%S")
    title = f"Training session {date} for {history.epoch[-1] + 1} epochs"

    if filename is None:
        filename = date

    for key, value in history.history.items():

        plt.plot(value, label=key)

    plt.title(title)
    plt.legend()
    plt.savefig(filename)


def _plot_history_altair(history, filename=None):

    data = history.history
    data['epoch'] = history.epoch
    data = pd.DataFrame(data)
    print(data)
    data = data.melt('epoch')
    print(data)
    chart = alt.Chart(data).mark_line().encode(x="epoch", y="value", color="variable")
    with open("./chart.png", "wb") as f:
        save(chart, f)


def dump_history(history, filename=None):

    date = datetime.datetime.now().strftime("%d-%m-%y-%H:%M:%S")

    history.model = None

    if filename is None:
        filename = date + ".json"

    with open(filename, "w") as f:
        json.dump(history.__dict__, f)
