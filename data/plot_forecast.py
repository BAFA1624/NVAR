import numpy as np
import matplotlib.pyplot as plt
import os

reconstruction = False
data_dir = os.path.dirname(os.path.abspath(__file__))

filename = ""
forecast_file = ""
test_file = ""

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
    reconstruction_data = {title: np.array(vals) for title, vals in reconstruction_data.items()}
# Load reconstruction labels
reconstruction_labels = {}
reconstruction_labels_filename = os.path.abspath(
    os.path.join(data_dir, "train_data", "21_0_-1_21000_0_1_1.csv")
)
with open(reconstruction_labels_filename, "r") as file:
    lines = file.readlines()
    coltitles = [s.strip() for s in lines[1].split(",")]
    reconstruction_labels = {title: [] for title in coltitles}
    for line in lines[2:]:
        vals = [np.float64(s.strip()) for s in line.split(",")]
        for title, x in zip(coltitles, vals):
            reconstruction_labels[title].append(x)
    reconstruction_labels = {title: np.array(vals) for title, vals in reconstruction_labels.items()} 

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

# Load test data
# 21_0_-1_21000_0_1_1
test_data = {}
test_file = "21_1_-1_189000_0_1_1.csv"
filename = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        data_dir, "test_data", test_file
    )
)
with open(filename, "r") as file:
    lines = file.readlines()
    coltitles = [s.strip() for s in lines[1].split(",")]
    test_data = {title: [] for title in coltitles}
    for line in lines[2:]:
        vals = [np.float64(s.strip()) for s in line.split(",")]
        for title, x in zip(coltitles, vals):
            test_data[title].append(x)
    test_data = {title: np.array(vals) for title, vals in test_data.items()}


# Plotting
fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


vmin = min(min(forecast_data["V"]), min(test_data["V"]))
vmin = max(-200, vmin)
vmax = max(max(forecast_data["V"]), max(test_data["V"]))
vmax = min(200, vmax)
ax1.plot(test_data["t"], test_data["V"], label="test_V")
ax1.plot(test_data["t"][-len(forecast_data["V"]):], forecast_data["V"], label="forecast_V")
ax1.set_ylim(vmin, vmax)
ax1.legend()

ax2.plot(test_data["t"], test_data["I"], label="test_I")
ax2.plot(test_data["t"][-len(forecast_data["I"]):], forecast_data["I"], label="forecast_I")
ax2.legend()

ax3.plot(reconstruction_labels["t"], reconstruction_labels["V"], label="True signal")
ax3.plot(reconstruction_labels["t"][-len(reconstruction_data["V"]):], reconstruction_data["V"], label="Reconstructed V")
ax3.legend()

ax4.plot(reconstruction_labels["t"], reconstruction_labels["I"], label="True signal")
ax4.plot(reconstruction_labels["t"][-len(reconstruction_data["I"]):], reconstruction_data["I"], label="Reconstructed I")
ax4.legend()


plt.show()
