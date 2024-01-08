import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os


# Setting global plotting parameters
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["legend.fontsize"] = 14


train_data = {}
filename = os.path.abspath(os.path.join(os.getcwd(), "train_data", "doublescroll.csv"))
with open(filename, "r") as file:
    lines = file.readlines()
    col_titles = [s.strip() for s in lines[0].split(",")]
    train_data = {title: [] for title in col_titles}
    for line in lines[1:]:
        vals = [np.float64(s.strip()) for s in line.split(",")]
        for title, x in zip(col_titles, vals):
            train_data[title].append(x)
    train_data = {title: np.array(vals) for title, vals in train_data.items()}

test_data = {}
filename = os.path.abspath(os.path.join(os.getcwd(), "test_data", "doublescroll.csv"))
with open(filename, "r") as file:
    lines = file.readlines()
    col_titles = [s.strip() for s in lines[0].split(",")]
    test_data = {title: [] for title in col_titles}
    for line in lines[1:]:
        vals = [np.float64(s.strip()) for s in line.split(",")]
        for title, x in zip(col_titles, vals):
            test_data[title].append(x)
    test_data = {title: np.array(vals) for title, vals in test_data.items()}

reconstruct_data = {}
filename = os.path.abspath(
    os.path.join(os.getcwd(), "forecast_data", "doublescroll_reconstruct.csv")
)
with open(filename, "r") as file:
    lines = list(file.readlines())
    col_titles = [s.strip() for s in lines[0].split(",")]
    reconstruct_data = {title: [] for title in col_titles}
    for line in lines[1:]:
        vals = [np.float64(s.strip()) for s in line.split(",")]
        for title, val in zip(col_titles, vals):
            reconstruct_data[title].append(val)
    reconstruct_data = {
        title: np.array(vals) for title, vals in reconstruct_data.items()
    }

forecast_data = {}
filename = os.path.abspath(
    os.path.join(os.getcwd(), "forecast_data", "doublescroll_predict.csv")
)
with open(filename, "r") as file:
    lines = list(file.readlines())
    col_titles = [s.strip() for s in lines[0].split(",")]
    forecast_data = {title: [] for title in col_titles}
    for line in lines[1:]:
        vals = [np.float64(s.strip()) for s in line.split(",")]
        for title, val in zip(col_titles, vals):
            forecast_data[title].append(val)
    forecast_data = {title: np.array(vals[:]) for title, vals in forecast_data.items()}

n_train = len(reconstruct_data[list(reconstruct_data.keys())[0]])
n_test = len(forecast_data[list(forecast_data.keys())[0]])

fig, (ax2, ax4, ax6) = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)

# Left column: Reconstructed training data
#ax1 = fig.add_subplot(321)  # v1
#ax3 = fig.add_subplot(323)  # v2
#ax5 = fig.add_subplot(325)  # I
# Right column: Forecast data

ax2.set_ylabel("V1 (Arbitrary Units)")
ax2.plot(forecast_data["t"], forecast_data["v1"], label="forecast_v1", marker="o", markersize=1, color="r", linestyle="")
ax2.plot(forecast_data["t"], forecast_data["v1'"], label="test_v1", linewidth=0.8, color="k")

ax4.set_ylabel("V2 (Arbitrary Units)")
ax4.plot(forecast_data["t"], forecast_data["v2"], label="forecast_v2", marker="o", markersize=1, color="r", linestyle="")
ax4.plot(forecast_data["t"], forecast_data["v2'"], label="test_v2", linewidth=0.8, color="k")

ax6.set_ylabel("I (Arbitrary Units)")
#ax6.set_ylim(max(forecast_data["I"])+0.1*max(forecast_data["I"]), min(forecast_data["I"]) + 0.1 * min(forecast_data["I"]))
ax6.plot(forecast_data["t"], forecast_data["I"], label="forecast_I", marker="o", markersize=1, color="r", linestyle="")
ax6.plot(forecast_data["t"], forecast_data["I'"], label="test_I", linewidth=0.8, color="k")
ax6.set_xlabel("Time (Arbitrary Units)")

plt.tight_layout()
plt.legend()
plt.show()
