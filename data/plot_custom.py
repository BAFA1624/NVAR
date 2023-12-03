import numpy as np
import matplotlib.pyplot as plt
import os

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

# Plotting
fig, axs = plt.subplots(
    len(forecast_keys), 2, constrained_layout=True, figsize=(20, 10)
)


for fkey, lkey, (ax1, ax2) in zip(forecast_keys, label_keys, axs):
    # Plot reconstruction of training data
    n = len(reconstruction_data[fkey])
    ax1.plot(
        reconstruction_labels["t"][-n:],
        reconstruction_data[fkey],
        label=fkey,
        linestyle="",
        marker="o",
        markersize=1,
        color="r",
    )
    ax1.plot(
        reconstruction_labels["t"][-n:],
        reconstruction_data[lkey],
        label="True Signal",
        linestyle="-",
        linewidth=0.7,
        color="k",
    )
    ax1.set_title(f"Reconstructed Training Data: {fkey}")
    ax1.legend()

    # Plot forecasted data
    ax2_max = min(100, max(forecast_data[fkey]))
    ax2_min = max(-100, min(forecast_data[fkey]))
    ax2.plot(
        times,
        forecast_data[fkey],
        label=fkey,
        linestyle="",
        marker="o",
        markersize=1,
        color="r",
    )
    ax2.plot(
        times,
        forecast_data[lkey],
        label=f"True {fkey}",
        linestyle="-",
        linewidth=0.7,
        color="k",
    )
    ax2.set_ylim(ax2_min, ax2_max)
    ax2.set_title(f"Forecast: {fkey}")
    ax2.legend()


plt.show()
