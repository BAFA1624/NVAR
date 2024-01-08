import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Setting global plotting parameters
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["legend.fontsize"] = 14


def parse_csv(filepath):
    data = None
    with open(filepath, "r") as file_obj:
        lines = file_obj.readlines()
        titleline, lines = lines[0], lines[1:]
        titles = [title.strip() for title in titleline.split(",")]
        data = {title: [] for title in titles}
        for line in lines:
            vals = [np.float64(s.strip()) for s in line.split(",")]
            for title, val in zip(titles, vals):
                data[title].append(val)
    return data


def get_voltage_histogram():
    return


BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), "distribution_data"))

PROTOCOLS = [17, 21, 23]
alt_types = ["default", "standardized", "normalized", "custom_standardized"]
data_types = ["measured", "integrated"]

FILE_PATHS = [
    os.path.join(BASE_PATH, f"{PROTOCOL}_{source_type}_{filetype}.csv")
    for PROTOCOL in PROTOCOLS
    for source_type in data_types
    for filetype in alt_types
    if os.path.exists(
        os.path.join(BASE_PATH, f"{PROTOCOL}_{source_type}_{filetype}.csv")
    )
]

print("Processing:")
for file in FILE_PATHS:
    print(f"\t- {file}")


csv_data = {
    PROTOCOL: {
        filetype: {
            source_type: parse_csv(
                os.path.join(BASE_PATH, f"{PROTOCOL}_{source_type}_{filetype}.csv")
            )
            for source_type in data_types
        }
        for filetype in alt_types
    }
    for PROTOCOL in PROTOCOLS
}

col1 = data_types[0]
col2 = data_types[1]

# Plot histograms

n = 17
nbins = 30

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(12, 4), sharey="row", constrained_layout=True
)

ax1.hist(csv_data[n]["default"]["measured"]["V"], bins=nbins, edgecolor="black")
ax1.set_xticks([-100, -50, 0, 50])
ax1.set_xlabel("Measured Voltage Distribution / mV")
ax1.set_ylabel("Frequency")

ax2.hist(csv_data[n]["normalized"]["measured"]["V"], bins=nbins, edgecolor="black")
ax2.set_xlabel("Normalized Voltage / mV")

ax3.hist(csv_data[n]["standardized"]["measured"]["V"], bins=nbins, edgecolor="black")
ax3.set_xlabel("Standardized Voltage / mV")
ax3.set_xticks([-2, 0, 2, 4])

plt.tight_layout()
plt.show()
