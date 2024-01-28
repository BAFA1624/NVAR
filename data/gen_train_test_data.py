import os
from sklearn.model_selection import KFold, TimeSeriesSplit, ShuffleSplit, RepeatedKFold
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import numpy as np


mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["legend.fontsize"] = 12


show_plots = True
save_figs = False


def plot_data(splits, xmin=None, xmax=None, show=True, filename=None, savefig=False):
    for i in splits:
        plt.plot(
            splits[i]["train"]["X"][:, 0]/100,
            splits[i]["train"]["X"][:, -1],
            linestyle="-",
            linewidth=0.8,
            label=f"{i}_train",
            #label="Training data",
            color="k",
        )
        plt.plot(
            splits[i]["test"]["X"][:, 0]/100,
            splits[i]["test"]["X"][:, -1],
            linestyle="-",
            linewidth=0.8,
            label=f"{i}_test",
            #label="Testing data",
            color="r",
        )
    if filename and not savefig:
        plt.title(filename)
    #plt.xlabel("Time / ms")
    #plt.ylabel("Measured Voltage / mV")
    plt.xlim(xmin, xmax)
    plt.legend()
    plt.tight_layout()
    if savefig and filename:
        path = os.path.join(os.getcwd(), "graphs", filename.split(".")[0])
        plt.savefig(path)
    if show:
        plt.show()

    return


def get_file_params(filename):
    params = {}

    pattern = re.compile(
        r"^(?P<file_N>[\d]+)_(?P<datatype>[012])"
        r"_(?P<n_points>[-\d]\d)_(?P<seed>[\d]+)"
        r"_(?P<file_type>[012])"
        r"_(?P<n>[\d]+)_(?P<N>[\d]+).csv$"
    )

    match = pattern.match(filename)
    if not match:
        return None

    for key in params.keys():
        params[key] = match.group(key)

    return params


def get_filename(
    file_N, train_test, seed, n_points, measured_integrated, n, N, filetype="csv"
):
    return f"{file_N}_{train_test}_{seed if seed else str(-1)}_{n_points}_{measured_integrated}_{n}_{N}.{filetype}"


def write_files(data, file_N, seed, measured_integrated, n, N, col_titles=None):
    keys = ["train", "test"]

    for key in keys:
        n_points = len(data[n][key]["indices"])
        train_test = 0 if key == "train" else 1

        filename = os.path.abspath(
            os.path.join(
                os.getcwd(),
                key + "_data",
                get_filename(
                    file_N, train_test, seed, n_points, measured_integrated, n + 1, N
                ),
            )
        )

        if not col_titles:
            col_titles = [str(i) for i in range(data[n][key]["X"].shape[1] + 1)]

        print(f"Writing: {filename}")
        with open(filename, "w") as file:
            # Writing header info
            file.write(
                f"file_N:{file_N},{key}"
                f",seed:{seed},percent:{n_points},"
                f"{'integrated' if measured_integrated else 'measured'},"
                f"n:{n+1},N:{N}\n"
            )
            assert (
                len(col_titles)
                == data[i][key]["X"].shape[1] + data[i][key]["y"].shape[1]
            )
            file.write(",".join(col_titles) + "\n")
            # Writing data
            for X, Y in zip(data[n][key]["X"], data[n][key]["y"][:, :]):
                vals = [str(x) for x in X] + [str(y) for y in Y]
                file.write(",".join(vals) + "\n")
            print(f"Wrote {filename}.")


def load_file(N, load_measured=True):
    src_dir = os.path.abspath(os.path.join(os.getcwd(), "train_test_src"))

    measured_filename = os.path.join(src_dir, N + "_measured.csv")
    integrated_filename = os.path.join(src_dir, N + "_integrated.csv")

    measured_data = {}
    if load_measured:
        with open(measured_filename, "r") as file:
            lines = file.readlines()
            col_titles = lines[0].strip().split(",")
            measured_data = {title: [] for title in lines[0].strip().split(",")}
            for line in lines[1:]:
                vals = [np.float64(s.strip()) for s in line.strip().split(",")]
                for title, x in zip(col_titles, vals):
                    measured_data[title].append(x)
        measured_data = {key: np.array(arr) for key, arr in measured_data.items()}

    integrated_data = {}
    with open(integrated_filename, "r") as file:
        lines = file.readlines()
        col_titles = lines[0].strip().split(",")
        integrated_data = {title: [] for title in lines[0].strip().split(",")}
        for line in lines[1:]:
            vals = [np.float64(s.strip()) for s in line.strip().split(",")]
            for title, x in zip(col_titles, vals):
                integrated_data[title].append(x)
    integrated_data = {key: np.array(arr) for key, arr in integrated_data.items()}

    if load_measured:
        return measured_data, integrated_data
    else:
        return integrated_data


def get_X_y(data):
    X = np.array([arr[0:-1] for arr in data.values()]).T
    y = None
    if "I" in data:
        y = np.array([data["I"][1:], data["V"][1:]]).T
    else:
        y = np.array([data["V"][1:]]).T
    return X, y


def k_split(data, n_splits=5, random_state=None):
    ksplit = KFold(n_splits=n_splits, shuffle=False, random_state=random_state)
    X, y = get_X_y(data)
    return ksplit.split(X, y)


def ts_split(data, n_splits=5, max_train_size=None, test_size=None, gap=0):
    tsplit = TimeSeriesSplit(
        n_splits=n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap
    )
    X, y = get_X_y(data)
    return tsplit.split(X, y)


def shuffle_split(data, n_splits=5, test_size=None, train_size=None, random_state=None):
    split = ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
    )
    X, y = get_X_y(data)
    return split.split(X, y)


def rk_split(data, n_splits=5, n_repeats=10, random_state=None):
    rksplit = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    X, y = get_X_y(data)
    return rksplit.split(X, y)


def simple_split(data, n_splits=1, train_fraction=0.1, train_size=None, init_offset=0):
    assert int(n_splits) > 0

    n = len(data["t"])

    test_size = int()
    if not train_size:
        assert 0 <= train_fraction < 1
        train_size = int((n / int(n_splits)) * train_fraction)
        test_size = int((n / int(n_splits)) * (1 - train_fraction))
    else:
        assert train_size < int(n / int(n_splits))
        test_size = int((n / int(n_splits))) - int(train_size + init_offset + 1)

    pos = init_offset
    while pos + train_size + test_size < n:
        train_idx = np.array(range(pos, pos + train_size))
        test_idx = np.array(range(pos + train_size, pos + train_size + test_size))
        pos += train_size + test_size
        yield train_idx, test_idx


source_dir = os.path.abspath(os.path.join(os.getcwd(), "train_test_src"))
files = os.listdir(source_dir)

pattern = re.compile(r"^(?P<N>[\d]+)_(?P<filetype>[\w]+).csv$")
file_nos = [17, *range(21, 31)]
cv = simple_split
cv_args = [1, 0.5]
for j, file in enumerate(files):
    match = pattern.match(file)
    if not match:
        continue

    N = match.group("N")
    measured_data, integrated_data = load_file(N)

    seed = None
    # seed = np.random.randint(10, 100)
    # cv_args = [5, None, None, seed]
    X, y = get_X_y(measured_data)
    measured_splits = {
        i: {
            "train": {
                "indices": train_idx,
                "X": X[train_idx, :],
                "y": y[train_idx, :],
            },
            "test": {
                "indices": test_idx,
                "X": X[test_idx, :],
                "y": y[test_idx, :],
            },
        }
        for i, (train_idx, test_idx) in enumerate(cv(measured_data, *cv_args))
    }

    X, y = get_X_y(integrated_data)
    integrated_splits = {
        i: {
            "train": {
                "indices": train_idx,
                "X": X[train_idx, :],
                "y": y[train_idx, :],
            },
            "test": {
                "indices": test_idx,
                "X": X[test_idx, :],
                "y": y[test_idx, :],
            },
        }
        for i, (train_idx, test_idx) in enumerate(cv(integrated_data, *cv_args))
    }

    measured_min = min(measured_data["t"])
    measured_max = max(measured_data["t"])
    plot_data(
        measured_splits,
        xmin=measured_min,
        xmax=measured_max,
        show=show_plots,
        filename=get_filename(
            N, 0, None, len(measured_data["t"]), 0, 0, cv_args[0], "png"
        ),
        savefig=save_figs,
    )
    integrated_min = min(integrated_data["t"])
    integrated_max = max(integrated_data["t"])
    plot_data(
        integrated_splits,
        xmin=integrated_min,
        xmax=integrated_max,
        show=show_plots,
        filename=get_filename(
            N, 0, None, len(integrated_data["t"]), 1, 0, cv_args[0], "png"
        ),
        savefig=save_figs,
    )

    for i in measured_splits:
        write_files(
            measured_splits, N, None, 0, i, cv_args[0], ["t", "I", "V", "I'", "V'"]
        )
    for i in integrated_splits:
        write_files(
            integrated_splits,
            N,
            None,
            1,
            i,
            cv_args[0],
            [
                "t",
                "V",
                "V'",
            ],
        )

# Fixed time step integrated data:
# file_nos = [31]
# cv = simple_split
# cv_args = [1, 0.1]
# for j, file in enumerate(file_nos):
#    match = pattern.match(file)
#    if not match:
#        continue
#
#    N = match.group("N")
#    integrated_data = load_file(N, load_measured=False)
#
#    seed = None
#    # seed = np.random.randint(10, 100)
#    # cv_args = [5, None, None, seed]
#    X, y = get_X_y(integrated_data)
#    integrated_splits = {
#        i: {
#            "train": {
#                "indices": train_idx,
#                "X": X[train_idx, :],
#                "y": y[train_idx, :],
#            },
#            "test": {
#                "indices": test_idx,
#                "X": X[test_idx, :],
#                "y": y[test_idx, :],
#            },
#        }
#        for i, (train_idx, test_idx) in enumerate(cv(integrated_data, *cv_args))
#    }
#
#    integrated_min = min(integrated_data["t"])
#    integrated_max = max(integrated_data["t"])
#    plot_data(
#        integrated_splits,
#        xmin=integrated_min,
#        xmax=integrated_max,
#        show=True,
#        filename=get_filename(N, 0, None, len(integrated_data["t"]), 1, 0, cv_args[0], "png"),
#        savefig=True,
#    )
#
#    for i in integrated_splits:
#        write_files(
#            integrated_splits,
#            N,
#            None,
#            1,
#            i,
#            cv_args[0],
#            [
#                "t",
#                "V",
#                "V'",
#            ],
#        )
