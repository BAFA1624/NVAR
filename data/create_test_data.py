import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os

dt = 0.25
warmup = 1.0
traintime = 100.0
testtime = 800.0
maxtime = warmup + traintime + testtime

warmup_pts = round(warmup / dt)
traintime_pts = round(traintime / dt)
testtime_pts = round(testtime / dt)
maxtime_pts = round(maxtime / dt)

t_eval = np.linspace(0, maxtime, maxtime_pts)

r1 = 1.2
r2 = 3.44
r4 = 0.193
alpha = 11.6
ir = 2 * 2.25e-5


def doublescroll(t, y):
    dV = y[0] - y[1]  # V1-V2
    g = (dV / r2) + ir * np.sinh(alpha * dV)
    dy0 = (y[0] / r1) - g
    dy1 = g - y[2]
    dy2 = y[1] - r4 * y[2]

    return [dy0, dy1, dy2]


rk23_sol = solve_ivp(
    doublescroll,
    (0, maxtime),
    [0.37926545, 0.058339, -0.08167691],
    t_eval=t_eval,
    method="RK23",
)
dop853_sol = solve_ivp(
    doublescroll,
    (0, maxtime),
    [0.37926545, 0.058339, -0.08167691],
    t_eval=t_eval,
    method="DOP853",
)

sol = dop853_sol

# Ground truth calculation
fp_slope = (r1 / r2) - (r4 / r2) - 1
fp_sinhx = alpha * (1 - r4 / r1)
fp_sinhy = r1 * ir
fp_xs = np.linspace(-1.4, 1.4, 50)
V1_fp, info, ier, mesg = fsolve(
    lambda V1: fp_slope * V1 + fp_sinhy * np.sinh(fp_sinhx * V1), 1.05, full_output=True
)
# setup variabled for predicted and true fixed points
t_fp0 = np.zeros(3)
t_fp1 = np.zeros(3)
t_fp2 = np.zeros(3)
# true fixed point 0 is 0
# true fixed point 1 is at...
t_fp1[0] = V1_fp[0]
t_fp1[1] = r4 * V1_fp[0] / r1
t_fp1[2] = V1_fp[0] / r1
# true fixed point 2 is at...
t_fp2[0] = -t_fp1[0]
t_fp2[1] = -t_fp1[1]
t_fp2[2] = -t_fp1[2]

filename = os.path.abspath(os.path.join(os.getcwd(), "train_data", "doublescroll.csv"))
with open(filename, "w") as file:
    file.write("t,v1,v2,i\n")
    vals = zip(sol.y[0][:400], sol.y[1][:400], sol.y[2][:400])
    for t, (v1, v2, i) in zip(sol.t, vals):
        file.write(",".join([str(t), str(v1), str(v2), str(i)]) + "\n")

filename = os.path.abspath(os.path.join(os.getcwd(), "test_data", "doublescroll.csv"))
with open(filename, "w") as file:
    file.write("t,v1,v2,i\n")
    vals = zip(sol.y[0][400:], sol.y[1][400:], sol.y[2][400:])
    for t, (v1, v2, i) in zip(sol.t, vals):
        file.write(",".join([str(t), str(v1), str(v2), str(i)]) + "\n")
