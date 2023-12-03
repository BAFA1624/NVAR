import numpy as np
import matplotlib.pyplot as plt
import os


def load_files(file_params):
    length = len(file_params["N"])
    for vals in list(file_params.values())[1:]:
        assert length == len(vals)

    data = {i: {} for i in range(length)}
    for i in range(length):
        filename = file_params["filenames"][i]
        skipheader = file_params["skipheader"][i]
        delim = file_params["delim"][i]

        with open(filename, "r") as file:
            lines = file.readlines()[skipheader:]
            titles = [title.strip() for title in lines[0].strip().split(delim)]
            data[i] = {title: [] for title in titles}
            for line in lines[1:]:
                vals = [np.float64(x.strip()) for x in line.split(delim)]
                for title, val in zip(titles, vals):
                    data[i][title].append(val)
            data[i] = {title: np.array(vals) for (title, vals) in data[i].items()}
    return data


def plot_data(data, file_params):
    for i in range(len(data)):
        xkey = file_params["xkey"][i]
        ykeys = file_params["ykeys"][i]
        xvals = data[i][xkey]
        for key in ykeys:
            label = (
                f"{i}_{key}"
                if not file_params["prepend_labels"][i]
                else f"{file_params['prepend_labels'][i]}_{key}"
            )
            plt.plot(xvals, data[i][key], markersize=0.01, marker="o", label=label)

    plt.legend()
    plt.show()


def get_filenames(data):
    length = len(data["N"])
    for vals in list(data.values())[1:]:
        assert len(vals) == length

    files = [
        f"{data['N'][i]}_{data['train_test'][i]}_{-1 if not data['seed'][i] else data['seed'][i]}_{data['n_points'][i]}_{data['measured_integrated'][i]}_{data['n'][i]}_{data['total_n'][i]}.csv"
        for i in range(length)
    ]
    current_dir = os.getcwd()
    directory_map = {0: "train_data", 1: "test_data", 2: "forecast_data"}
    data["filenames"] = [
        os.path.join(current_dir, directory_map[train_test], file)
        for train_test, file in zip(data["train_test"], files)
    ]
    return data


# Filename parameters
file_params = {
    "N": [17, 17, 17, 17],
    "train_test": [0, 1, 0, 1],
    "seed": [None, None, None, None],
    "n_points": [10000, 95499, 10000, 95499],
    "measured_integrated": [0, 0, 0, 0],
    "n": [1, 1, 2, 2],
    "total_n": [2, 2, 2, 2],
    "delim": [",", ",", ",", ","],
    "skipheader": [1, 1, 1, 1],
    "xkey": ["t", "t", "t", "t"],
    "ykeys": [
        # ["I", "V", "I'", "V'"],
        # ["I", "V", "I'", "V'"],
        ["V"],
        ["V"],
        ["V"],
        ["V"],
    ],
    "prepend_labels": ["train_1", "test_1", "train_2", "test_2"],
}

file_params = get_filenames(file_params)

data = load_files(file_params)

plot_data(data, file_params)
