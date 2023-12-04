import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb


plot = True
savefigs = False
graph_loc = os.path.abspath(os.path.join(os.getcwd(), "graphs"))


def decompress(file_loc, file):
    def transform_signals(signals):
        return np.array([signals[:, 0] * 0.1, signals[:, 1] * 5])

    record = wfdb.rdrecord(os.path.join(file_loc, file))
    col_titles = record.sig_name
    print(col_titles)
    signals = transform_signals(record.p_signal)
    dt = 1 / record.fs
    times = dt * np.array(list(range(record.sig_len)))

    return times, {title: signal for title, signal in zip(col_titles, signals)}


def get_files(search_dir):
    filenames = os.listdir(search_dir)

    signal_files = set()

    for name in filenames:
        signal_files.add(os.path.basename(name).split(".")[0])

    return list(sorted(signal_files))


search_dir = os.path.abspath(os.path.join(os.getcwd(), "hh_data"))

signal_files = get_files(search_dir)

#for file in signal_files:
for file in ["a2t11", "a4t15", "a6t12", "a6t38", "a8t19"]:
    times, data = decompress(search_dir, file)

    fig, axs = plt.subplots(len(data), 1, constrained_layout=True, figsize=(12, 8))

    for title, ax in zip(data.keys(), axs):
        n = int(0.33 * len(data[title]))
        ax.axvline(times[n])
        ax.plot(times, data[title], linestyle="", marker="o", markersize=1, color="k")
        ax.set_title(title)

    fig.suptitle(file)

    if savefigs:
        fig.savefig(os.path.join(graph_loc, file))
    if plot:
        plt.show()
    else:
        plt.close()

    # Remove file
    os.remove(os.path.join(search_dir, file))
