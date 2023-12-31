import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = os.path.dirname(os.path.abspath(__file__))

# Load forecast data
forecast_data = {}
forecast_path = os.path.abspath(
    os.path.join(data_dir, "forecast_data", "esn_forecast.csv")
)
with open(forecast_path, "r") as file:
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

if len(forecast_keys) > 1:
    # Plotting
    fig, axs = plt.subplots(
        len(forecast_keys), 1, constrained_layout=True, figsize=(20, 10)
    )
    for fkey, lkey, ax1 in zip(forecast_keys, label_keys, axs):
        # Plot reconstruction of training data
        print(fkey, lkey)

        # Plot forecasted data
        ax1min = min(forecast_data[lkey])
        ax1max = max(forecast_data[lkey])
        ax1.plot(
            #times,
            forecast_data[fkey],
            label=fkey,
            linestyle="",
            marker="o",
            markersize=1,
            color="r",
        )
        ax1.plot(
            #times,
            forecast_data[lkey],
            label=f"True {fkey}",
            linestyle="-",
            linewidth=0.7,
            color="k",
        )
        ax1.set_ylim(ax1min - 0.1 * abs(ax1min), ax1max + 0.1 * abs(ax1max))
        ax1.set_title(f"Forecast: {fkey}")
        ax1.legend()
else:
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    ax1 = fig.add_subplot(121)

    fkey = forecast_keys[0]
    lkey = label_keys[0]

    ax1min = min(forecast_data[lkey])
    ax1max = max(forecast_data[lkey])
    ax1.plot(
        #times,
        forecast_data[fkey],
        label=fkey,
        linestyle="",
        marker="o",
        markersize=1,
        color="r",
    )
    ax1.plot(
        #times,
        forecast_data[lkey],
        label=f"True {fkey}",
        linestyle="-",
        linewidth=0.7,
        color="k",
    )
    ax1.set_ylim(ax1min - 0.1 * abs(ax1min), ax1max + 0.1 * abs(ax1max))
    ax1.set_title(f"Forecast: {fkey}")
    ax1.legend()

plt.show()
