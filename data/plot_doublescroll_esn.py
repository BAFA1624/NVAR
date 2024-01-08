import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Setting global plotting parameters
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["legend.fontsize"] = 14


data_dir = os.path.dirname(os.path.abspath(__file__))

# Load forecast data
forecast_data = {}
forecast_path = os.path.abspath(
    os.path.join(data_dir, "forecast_data", "esn_doublescroll_forecast.csv")
)
with open(forecast_path, "r") as file:
    lines = file.readlines()
    coltitles = [s.strip() for s in lines[0].split(",")]
    forecast_data = {title: [] for title in coltitles}
    for line in lines[1:]:
        vals = [np.float64(s.strip()) for s in line.split(",")]
        for title, x in zip(coltitles, vals):
            forecast_data[title].append(x)
    forecast_data = {title: np.array(vals[:]) for title, vals in forecast_data.items()}

times = forecast_data["t"]

fig = plt.figure(figsize=(10, 8), constrained_layout=True)

gs = GridSpec(3, 1)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

ax1.plot(
    times,
    forecast_data["I"],
    color="r",
    linestyle="",
    marker="o",
    markersize=1,
    label="I",
)
ax1.plot(
    times,
    forecast_data["I'"],
    color="k",
    linestyle="-",
    linewidth=0.7,
)
ax1.set_xlabel("Time (Arbitrary Units)")
ax1.set_ylabel("I (Arbitrary Units)")

ax2.plot(
    times,
    forecast_data["V1"],
    color="r",
    linestyle="",
    marker="o",
    markersize=1,
    label="V1",
)
ax2.plot(
    times,
    forecast_data["V1'"],
    color="k",
    linestyle="-",
    linewidth=0.7,
)
ax2.set_xlabel("Time (Arbitrary Units)")
ax2.set_ylabel("V1 (Arbitrary Units)")

ax3.plot(
    times,
    forecast_data["V2"],
    color="r",
    linestyle="",
    marker="o",
    markersize=1,
    label="V2",
)
ax3.plot(
    times,
    forecast_data["V2'"],
    color="k",
    linestyle="-",
    linewidth=0.7,
)
ax3.set_xlabel("Time (Arbitrary Units)")
ax3.set_ylabel("V2 (Arbitrary Units)")

plt.tight_layout()
plt.show()
