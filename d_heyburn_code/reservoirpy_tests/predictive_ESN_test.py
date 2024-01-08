import os
import pandas as pd
import reservoirpy as rpy
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge, ESN
from reservoirpy.observables import nrmse

"""
In this programme we explore predicting voltage signals by training the data on injected current and on previous 
voltage.We keep most parameters consistent, with the hope that the additional context provided to the model will 
allow us to reduce the observation window.

All data has already had their transient states removed, after which both the current and the voltage signals are 
normalised. 
"""

# Setup:
SEED = 42  # Repeatability
VERBOSE = True  # Provides information about the model training
data_dict = os.path.join(os.path.dirname(os.getcwd()), "data", "Normalised_Signals")
TRAIN_SIZE = 15000
TEST_SIZE = 100000
HORIZON = 1
FEEDBACK = False

# Reservoir params:
input_bias = True
units = 500  # Number of recurrent units in the reservoir.
leak_rate = 0.8
rho = 1.00  # The spectral radius of the reservoir weight matrix.
input_scaling = 1.0
rc_connectivity = 0.1
in_connectivity = 1.0
fb_connectivity = 1.0

# Readout params:
warmup = 100
regularizer = 1e-6

if __name__ == "__main__":

    rpy.set_seed(seed=SEED)
    rpy.verbosity(1)

    datafile = os.path.join(data_dict, "M_NProtocol_17")
    signal = pd.read_csv(datafile, header=0, delimiter=",", dtype="float").values

    datafile2 = os.path.join(data_dict, "M_NProtocol_21")
    signal2 = pd.read_csv(datafile, header=0, delimiter=",", dtype="float").values

    # Print out dataset properties for review
    print(f"Signal type: {type(signal)}")
    print(f"Signal shape: {signal.shape}")

    # ----- Train/Test Split -----
    x_train = signal[0:TRAIN_SIZE, :]  #.reshape(-1, 1)
    print(x_train.shape)
    y_train = signal[1:TRAIN_SIZE+HORIZON, -1].reshape(-1, 1)
    print(y_train.shape)

    x_test = signal2[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE, 0].reshape(-1, 1)
    print(x_test.shape)
    y_test = signal2[TRAIN_SIZE+HORIZON:TRAIN_SIZE+TEST_SIZE+HORIZON, 1].reshape(-1, 1)
    print(y_test.shape)

    # ----- Reservoir Setup -----
    reservoir = Reservoir(
        units,
        lr=leak_rate,
        sr=rho,
        input_bias=input_bias,
        input_scaling=input_scaling,
        rc_connectivity=rc_connectivity,
        input_connectivity=in_connectivity,
        fb_connectivity=fb_connectivity,
        name="reservoir"
    )

    # ----- Readout Setup -----
    readout = Ridge(
        ridge=regularizer,
        name="readout"
    )

    if FEEDBACK:
        reservoir = reservoir << readout

    # ---- ESN Setup -----
    esn = ESN(
        reservoir=reservoir,
        readout=readout,
        workers=-1
    )

    # ----- Run Experiment -----
    internal_states = esn.fit(X=x_train, Y=y_train, warmup=100)
    y_pred = esn.run(x_test)

    # Accuracy assessment:
    error = nrmse(y_test, y_pred)
    print(f"NRMSE = {error}")

    plt.figure(figsize=(10, 4))
    plt.plot(y_pred, color="r", lw=1.5, label="Prediction")
    plt.plot(y_test, color="k", lw=1.0, label="Ground truth")
    plt.title("Forecasted Membrane Voltage (Integrated)")
    plt.legend()

    plt.savefig("V_measured_forecast.png")
    plt.show()
