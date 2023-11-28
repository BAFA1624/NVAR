import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os

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
filename = os.path.abspath(os.path.join(os.getcwd(), "forecast_data", "doublescroll_reconstruct.csv"))
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
    forecast_data = {title: np.array(vals) for title, vals in forecast_data.items()}

n_train = len(reconstruct_data[list(reconstruct_data.keys())[0]])
n_test = len(forecast_data[list(forecast_data.keys())[0]])

fig = plt.figure(figsize=(8, 8))

# Left column: Reconstructed training data
ax1 = fig.add_subplot(321) # v1
ax3 = fig.add_subplot(323) # v2
ax5 = fig.add_subplot(325) # I
# Right column: Forecast data
ax2 = fig.add_subplot(322) # v1
ax4 = fig.add_subplot(324) # v2
ax6 = fig.add_subplot(326) # I

ax1.set_title("Reconstructed training data\nV1")
ax1.plot(train_data["t"][-n_train:], reconstruct_data["v1"], label="reconstructed_v1")
ax1.plot(train_data["t"][-n_train:], train_data["v1"][-n_train:], label="train_v1")

ax3.set_title("V2")
ax3.plot(train_data["t"][-n_train:], reconstruct_data["v2"], label="reconstructed_v2")
ax3.plot(train_data["t"][-n_train:], train_data["v2"][-n_train:], label="train_v2")

ax5.set_title("I")
ax5.plot(train_data["t"][-n_train:], reconstruct_data["I"], label="reconstructed_I")
ax5.plot(train_data["t"][-n_train:], train_data["i"][-n_train:], label="train_I")

ax2.set_title("Forecast data\nV1")
ax2.plot(test_data["t"][-n_test:], forecast_data["v1"], label="forecast_v1")
ax2.plot(test_data["t"][-n_test:], test_data["v1"][-n_test:], label="test_v1")

ax4.set_title("V2")
ax4.plot(test_data["t"][-n_test:], forecast_data["v2"], label="forecast_v2")
ax4.plot(test_data["t"][-n_test:], test_data["v2"][-n_test:], label="test_v2")

ax6.set_title("I")
ax6.set_ylim(-3, 3)
ax6.plot(test_data["t"][-n_test:], forecast_data["I"], label="forecast_I")
ax6.plot(test_data["t"][-n_test:], test_data["i"][-n_test:], label="test_I")

plt.tight_layout()
plt.legend()
plt.show()
