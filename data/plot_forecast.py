import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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


data_stride = 3
data_dir = os.path.dirname(os.path.abspath(__file__))

# Load reconstruction data
reconstruction_data = {}
reconstruction_filename = os.path.abspath(
    os.path.join(data_dir, "forecast_data", "tmp.csv")
)
with open(reconstruction_filename, "r") as file:
    lines = file.readlines()
    coltitles = [s.strip() for s in lines[0].split(",")]
    reconstruction_data = {title: [] for title in coltitles}
    for line in lines[1:]:
        vals = [np.float64(s.strip()) for s in line.split(",")]
        for title, x in zip(coltitles, vals):
            reconstruction_data[title].append(x)
    reconstruction_data = {
        title: np.array(vals) for title, vals in reconstruction_data.items()
    }
# Load reconstruction labels
reconstruction_labels_raw = {}
reconstruction_labels_filename = os.path.abspath(
    os.path.join(data_dir, "train_data", "21_0_-1_21000_0_1_1.csv")
)
with open(reconstruction_labels_filename, "r") as file:
    lines = file.readlines()
    coltitles = [s.strip() for s in lines[1].split(",")]
    reconstruction_labels_raw = {title: [] for title in coltitles}
    for line in lines[2:]:
        vals = [np.float64(s.strip()) for s in line.split(",")]
        for title, x in zip(coltitles, vals):
            reconstruction_labels_raw[title].append(x)
    reconstruction_labels_raw = {
        title: np.array(vals) for title, vals in reconstruction_labels_raw.items()
    }

reconstruction_labels = {
    "t": reconstruction_labels_raw["t"][1::data_stride],
    "V_(n)": reconstruction_labels_raw["V"][1::data_stride],
    "V_(n-1)": reconstruction_labels_raw["V"][0:-1:data_stride],
    "I_(n)": reconstruction_labels_raw["I"][1::data_stride],
    "I_(n-1)": reconstruction_labels_raw["I"][0:-1:data_stride],
}

# Load forecast data
forecast_data = {}
forecast_filename = os.path.abspath(
    os.path.join(data_dir, "forecast_data", "forecast.csv")
)
with open(forecast_filename, "r") as file:
    lines = file.readlines()
    coltitles = [s.strip() for s in lines[0].split(",")]
    forecast_data = {title: [] for title in coltitles}
    for line in lines[1:]:
        vals = [np.float64(s.strip()) for s in line.split(",")]
        for title, x in zip(coltitles, vals):
            forecast_data[title].append(x)
    forecast_data = {title: np.array(vals) for title, vals in forecast_data.items()}

forecast_keys = [key for key in forecast_data.keys() if (key != "t" and key[-1] != "'")]
label_keys = [key for key in forecast_data.keys() if key[-1] == "'"]

print(f"forecast_keys: {forecast_keys}")
print(f"label_keys: {label_keys}")
assert len(forecast_keys) == len(label_keys)

times = forecast_data["t"]

fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))

#ax1.plot(times, forecast_data["I"], linestyle="", marker="o", markersize=1, color="r")
#ax1.plot(times, forecast_data["I'"], linestyle="-", linewidth=0.8, color="k")
#ax1.set_ylabel("Standardized Current / mA")
ax2.plot(times/100, forecast_data["V"], linestyle="", marker="o", markersize=1, color="r")
ax2.plot(times/100, forecast_data["V'"], linestyle="-", linewidth=0.8, color="k")
ax2.set_xlabel("Time / ms")
ax2.set_ylabel("Standardized Voltage / mV")
ymin = min(forecast_data["V'"]) - 0.1 * abs(min(forecast_data["V'"]))
ymax = max(forecast_data["V'"]) + 0.1 * abs(max(forecast_data["V'"]))
ax2.set_ylim(ymin, ymax)

plt.tight_layout()
plt.show()
