import matplotlib.pyplot as plt
import numpy as np
import os
import json


def load_file(file):
    data = {}
    with open(file, "r") as fp:
        lines = fp.readlines()
        col_titles = [s.strip() for s in lines[0].split(",")]
        data = {title: [] for title in col_titles}
        for line in lines[1:]:
            vals = [np.float64(s.strip()) for s in line.split(",")]
            for title, val in zip(col_titles, vals):
                data[title].append(val)
    data = {title: np.array(arr) for title, arr in data.items()}
    return data


file = open("../metadata/results_files.json", "r")
files_json = json.load(file)
file.close()

files = files_json["results_files"]

for file in files:
    # Load data
    data = load_file(file)
    keys = [key for key in data.keys() if key != "t" and key[-1] != "'" ]
    label_keys = [key for key in data.keys() if key != "t" and key[-1] == "'"]

    fig, axs = plt.subplots(int((len(data) - 1)/2), 1, constrained_layout=True, figsize=(12,8))
    for ax, key, lkey in zip(axs, keys, label_keys):
        labelmax, labelmin = max(data[lkey]), min(data[lkey])
        ymax = labelmax + abs(0.1 * labelmax)
        ymin = labelmin - abs(0.1 * labelmin)

        ax.set_ylim(ymin, ymax)
        ax.plot(data["t"], data[key], linestyle="", marker="o", markersize=1, color="k")
        ax.plot(data["t"], data[lkey], linestyle="", marker="x", markersize=1, color="r")
        ax.set_title(key)
    fig.suptitle(os.path.basename(file))
    plt.show()
    
    # Remove file
    os.remove(os.path.join(file))
