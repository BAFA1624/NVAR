import os
import re
import numpy as np
import matplotlib.pyplot as plt

show_plots = True
transient_sz = 20000

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
    print(N, len(it_data), len(mv_data) - transient_sz)
    times = np.linspace(min(it_data), max(it_data), len(mv_data) - transient_sz)

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

    if show_plots:
        figure_name = os.path.join(os.getcwd(), "graphs", f"{N}_measured_v_integrated")
        plt.title(str(N))
        plt.plot(
            times,
            mv_data[transient_sz:],
            marker=",",
            linestyle=None,
            label="measured",
        )
        #plt.plot(it_data, iv_data, linestyle="--", linewidth=0.8, label="integrated")
        plt.legend()
        plt.savefig(figure_name)
        plt.show()
