import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["legend.fontsize"] = 12

save_figs = False
show_plots = True
save_data = False
transient_sz = 19000

src_dir = os.path.abspath(os.path.join(os.getcwd(), "src_files"))
dir = os.path.abspath(os.path.join(os.getcwd(), "train_test_src"))

files = os.listdir(src_dir)

patterns = [
    ("measured_voltage", re.compile(r"^hhVsi_20110509_2_1_(?P<N>\d+)\.(?:dat|DAT)$")),
    ("measured_current", re.compile(r"^hhIsi_20110509_2_1_(?P<N>\d+)\.(?:dat|DAT)$")),
    ("integrated_voltage", re.compile(r"^V_int_20110509_2_1_(?P<N>\d+)\.(?:dat|DAT)$")),
]

file_types = {
    "measured_voltage": {},
    "measured_current": {},
    "integrated_voltage": {},
    "N": set(),
}

for file in files:
    for name, pattern in patterns:
        match = pattern.match(file)
        if match:
            N = match.group("N")
            file_types["N"].add(N)
            file_types[name][N] = os.path.join(src_dir, file)

# Map from fixed step files to correct measured file
fixed_step_N = {"31": "30", "32": "30"}
for N in sorted(file_types["N"]):
    mv_data = []
    measured_N = N
    if N in fixed_step_N:
        measured_N = fixed_step_N[N]

    with open(file_types["measured_voltage"][measured_N], "r") as file:
        lines = file.readlines()
        for line in lines:
            mv_data.append(np.float64(line.strip()))
    mi_data = []
    with open(file_types["measured_current"][measured_N], "r") as file:
        lines = file.readlines()
        for line in lines:
            mi_data.append(np.float64(line.strip()))
    it_data = []
    iv_data = []
    with open(file_types["integrated_voltage"][measured_N], "r") as file:
        lines = file.readlines()
        for line in lines:
            svals = line.strip().split("\t")
            it_data.append(np.float64(svals[0].strip()))
            iv_data.append(np.float64(svals[1].strip()))
    print(N, len(mv_data) - transient_sz, min(it_data), max(it_data), max(it_data) - min(it_data))
    times = np.linspace(min(it_data), max(it_data), len(mv_data) - transient_sz)

    if save_data:
    # Writing measured data to combined file
        filename = os.path.join(dir, f"{N}_measured.csv")
        with open(filename, "w") as file:
            lines = ["t,I,V\n"]
            for t, i, v in zip(times, mi_data[transient_sz:], mv_data[transient_sz:]):
                lines.append(",".join([str(t), str(i), str(v)]) + "\n")
            file.writelines(lines)

        # Writing integrated data to combined file
        filename = os.path.join(dir, f"{N}_integrated.csv")
        with open(filename, "w") as file:
            lines = ["t,V\n"]
            for (
                t,
                v,
            ) in zip(it_data, iv_data):
                lines.append(",".join([str(t), str(v)]) + "\n")
            file.writelines(lines)

    figure_name = os.path.join(os.getcwd(), "graphs", f"{N}_measured_v_integrated")

    #fig = plt.figure(figsize=(12, 4))
    #ax1 = fig.add_subplot(111)
    #ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)
    # plt.title(str(int(N)))
    # plt.title("Measured / Integrated Membrane Potential")

    fig = plt.figure(figsize=(16, 8))

    gs = GridSpec(3, 1)

    ax1 = fig.add_subplot(gs[0:2,:])
    ax2 = fig.add_subplot(gs[2:,:])

    ax1.plot(times / 100, mv_data[transient_sz:], linestyle="-", color="b", linewidth=0.8)
    ax2.plot(times / 100, mi_data[transient_sz:], linestyle="-", color="r", linewidth=0.8)
    
    ax1.set_ylabel("Measured Voltage / mV")
    ax2.set_xlabel("Time / ms")
    ax2.set_ylabel("Injected current / mA")
    
    ax1.set_xlim(0, 0.7)
    ax2.set_xlim(0, 0.7)

    #min = 13.5
    #max = 14.7
    #ax1.plot(
    #    (times / 100) - min,
    #    mv_data[transient_sz:],
    #    linestyle="-",
    #    label="Measured Voltage",
    #    color="k",
    #    linewidth=0.8,
    #)
    #ax1.plot(
    #    times / 100,
    #    mv_data[transient_sz:],
    #    linestyle="-",
    #    label="Injected Current",
    #    color="k",
    #    linewidth=0.8,
    #)
    #dt_ = max - min
    #ax1.set_xlim(0, dt_)
    #ax2.set_xlim(0, 10)
    #ax1.set_ylim(-50, 45)
    #ax1.set_xlabel("Injected Current / mA", fontname="Times New Roman", fontsize=18)
    #ax1.set_ylabel("Measured Voltage / mV", fontname="Times New Roman", fontsize=18)
    #ax2.set_xlabel("Time / ms", fontname="Times New Roman", fontsize=14)
    #ax2.set_ylabel("Injected Current / mA", fontname="Times New Roman", fontsize=14)

    # plt.plot(
    #    it_data, iv_data, linestyle="--", linewidth=0.8, label="integrated", color="r"
    # )
    if save_figs:
        plt.savefig(figure_name, dpi=200)
    if show_plots:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
