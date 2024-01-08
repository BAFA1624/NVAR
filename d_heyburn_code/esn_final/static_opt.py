import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A more optimal function for matrix inversions.
from scipy import linalg
# Importing functions used for random weight initialization.
from WeightsInit import *
# Importing functions for calculating model performance metrics.
from ErrorFunctions import *


"""
A static implementation of an ESN in python. Only the code near the top needs to be edited to change the parameters
of the reservoir. This is by no means the most optimal means to set up an ESN, but allows the user to see directly what
is being changed depending on the initial parameters. 
For hyperparameter tuning to measure the relationships between error metrics and model parameters, it is advised to use
more optimal implementations such as ReservoirPy or to implement the ESN in a lower level language. This file is to 
merely demonstrate how a generative or predictive ESN can be implemented for HVC neurons. 

In here, the model is trained by being fed the current injected into the neuron and it's simultaneous voltage, and is
taught to predict the next value for voltage. In the predictive phase, the model is given only the current and the 
initial value for voltage. From there, it must infer the voltage at each timestep and propagate its predictions.

All datasets should be of shape (n_dimensions, timesteps), this is faster than having to regularly transpose the signals
as each state update. As such, reshaping the data is required. Since the data are read in as csv files, we use pandas to
normalize or standardize as they can apply easy column-wise pre-processing.
"""


# ----- SOME FUNCTIONS FOR PRE-PREPROCESSING -----

def normalize(data: pd.DataFrame) -> pd.DataFrame:
    return (data - data.min()) / (data.max() - data.min())


def standardize1(data: pd.DataFrame) -> pd.DataFrame:
    return (data - data.mean()) / data.std()


def standardize2(data: pd.DataFrame) -> pd.DataFrame:
    return data / data.std()


def TTS_procedure(data: np.ndarray, train_len: int, test_len: int, warmup: len, dt: int) -> tuple:
    """
    Performs train/test split on the supplied data for timeseries prediction.

    :param data: A (t, K) matrix which are the linear features of the signal.
    :param train_len: The number of timesteps in our training data.
    :param test_len: The number of timesteps we wish to forecast ahead.
    :param warmup: The number of initial states to discard for when readout training begins. Fairly arbitrary.
    :param dt: The number of steps ahead the model is trained to predict. We default to 1 in our experiments.

    :return:
        u_train: Training data with shape (K, TRAIN).
        y_train: Training labels with shape (K, TRAIN-WARMUP).
        I_test: The driving force function with shape (1, TEST).
        V_test: The true values for voltage with shape (1, TEST).
    """

    # For training, we want all columns and as many timesteps as desired.
    u_train = data[0:train_len, :]
    y_train = data[warmup+dt:train_len+dt, 1].reshape(-1, 1)

    # For testing, we split the data into two 1D arrays. On for the driving current, and another for the true voltages.
    u_test = data[train_len:train_len+test_len, :]
    I_test = u_test[:, 0].reshape(-1, 1)
    V_test = u_test[:, 1].reshape(-1, 1)

    # The last values for u and state vector x are produced in the script. As such, all forecasting is currently a
    # continuation from the training data.

    return u_train.T, y_train.T, I_test.T, V_test.T

# STATIC OPTIONS (Never change)

# Set to true to enable figures and printing operations.
VERBOSE = False
# Temporal Parameters:
TRAIN = 75000
TEST = 25000
WARMUP = 0
DT = 1
# Pre-processing choice: Set both to 'False' to carry out no pre-processing.
# We've found that for this data, the model only works if it is standardized.
NORMALIZE = False
STANDARDIZE1 = False
STANDARDIZE2 = True
if sum([NORMALIZE, STANDARDIZE1, STANDARDIZE2]) != 1:
    raise ValueError("Only choose one preprocessing type.")
# Random distribution: Only one should be true at runtime.
ORDERED = False
UNIFORM = True
if sum([ORDERED, UNIFORM]) != 1:
    raise ValueError("Only chose one distribution type.")
# Echo-State Network Parameters:
# Input vector (K) size:
K = 2
# Output vector (L) size:
L = 1
# Miscellaneous parameters:
SEED = np.array(range(0, 21, 1))
# np.random.seed(seed=SEED)
DTYPE = "float32"  # Keep this as a float, but it is fine to change its size.
BIAS = False
current_date_time = time.asctime().replace(' ', '_').replace(':', '_')
OUTPUT_DIR = os.path.abspath(
    os.path.join(".", "metadata", current_date_time)
)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(OUTPUT_DIR):
    raise ValueError("Output directory must exist.")

# VARIABLE PARAMETERS

# Picking which current protocol to use. Options are: 17, 21-30.
PROTOCOL_values = np.array([17, 21, 23])
# PROTOCOL_values = np.array([17])
# Number of nodes (N) inside the reservoir:
N_values = np.array([1500])
# The leaking rate (a), the fraction of the previous state preserved at each step:
# a_values = np.round(np.linspace(0.7, 1, 6), 2)
a_values = np.array([0.9])
print(f"Leak rate search: {a_values}")
# The spectral radius (p), amplifies or dampens the chaotic nature of the signal. <1 is typical:
# p_values = np.round(np.linspace(0.9, 1.3, 6), 2)
p_values = np.array([1.06])
print(f"Spectral radius search: {p_values}")
# Input scaling for the input vector u:
# iss_values = np.round(np.linspace(0.4, 1.2, 6), 2)
iss_values = np.array([1.04])
print(f"Input scaling search: {iss_values}")
# Regularization for the L2 regression:
reg_values = np.array([0.0004])
# Connectivity: The probability of any two nodes being connected. Including self-connections.
in_connectivity_values = np.array([1.0])
rc_connectivity_values = np.array([0.003])

# ----- BEGIN PROGRAM -----

for PROTOCOL in PROTOCOL_values:
    results = {"results_Analysis": [], "order": []}
    for seed in SEED:
        for N in N_values:
            for a in a_values:
                for p in p_values:
                    for iss in iss_values:
                        for reg in reg_values:
                            for in_connectivity in in_connectivity_values:
                                for rc_connectivity in rc_connectivity_values:
                                    np.random.seed(seed=seed)

                                    result = {
                                        "params": {
                                            "N": str(N),
                                            "a": a,
                                            "p": p,
                                            "iss": iss,
                                            "reg": reg,
                                            "in_connectivity": in_connectivity,
                                            "rc_connectivity": rc_connectivity,
                                            "seed": str(seed)
                                        },
                                        "metrics": {}
                                    }
                                    print(result)

                                    start = time.monotonic()
                                    # ----- IMPORTING DATASET -----
                                    parent_dict = os.path.dirname(os.getcwd())
                                    # In the protocol string: M=Measured, I=Integrated.
                                    data_loc = os.path.join(parent_dict, "data", "Signals", f"M_Protocol_{PROTOCOL}")
                                    # Leave as a DataFrame to allow for column-wise pre-processing.
                                    data = pd.read_csv(data_loc, header=0, delimiter=',', dtype=DTYPE)[:200000:2]
                                    # We don't change the variable name here to allow no pre-processing if desired.
                                    if NORMALIZE:
                                        data = normalize(data=data).values
                                    elif STANDARDIZE1:
                                        data = standardize1(data=data).values
                                    elif STANDARDIZE2:
                                        data['Voltage'] = data['Voltage'] - data['Voltage'].mean()
                                        data = standardize2(data=data).values
                                    else:
                                        data = data.values

                                    # Plotting the train and test signals.
                                    if VERBOSE:
                                        print(data.shape)
                                        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
                                        timelist = range(TRAIN+TEST)

                                        # Plotting the injected current for the training part of the signal.
                                        axs[0].set_title("Injected Current")
                                        if NORMALIZE:
                                            axs[0].set_ylabel("Normalized Current")
                                        elif STANDARDIZE1 or STANDARDIZE2:
                                            axs[0].set_ylabel("Standardized Current")
                                        else:
                                            axs[0].set_ylabel("Current (mA)")
                                        # Warmup current
                                        axs[0].plot(timelist[:WARMUP], data[:WARMUP, 0], color="r", label="Warmup")
                                        # Training current
                                        axs[0].plot(timelist[WARMUP:TRAIN], data[WARMUP:TRAIN, 0], color="k", label="Train")
                                        # Testing current
                                        axs[0].plot(timelist[TRAIN:TRAIN+TEST], data[TRAIN:TRAIN + TEST, 0], color='b', label="Test")

                                        # Plotting the Voltage signal.
                                        axs[1].set_title("Injected Voltage")
                                        axs[1].set_xlabel("Time (ms)")
                                        if NORMALIZE:
                                            axs[1].set_ylabel("Normalized Voltage")
                                        elif STANDARDIZE1 or STANDARDIZE2:
                                            axs[1].set_ylabel("Standardized Voltage")
                                        else:
                                            axs[1].set_ylabel("Voltage (mV)")
                                        # Warmup current
                                        axs[1].plot(timelist[:WARMUP], data[:WARMUP, 1], color="r", label="Warmup")
                                        # Training voltage
                                        axs[1].plot(timelist[WARMUP:TRAIN], data[WARMUP:TRAIN, 1], color="k", label="Train")
                                        # Testing voltage
                                        axs[1].plot(timelist[TRAIN:TRAIN+TEST], data[TRAIN:TRAIN + TEST, 1], color='b', label="Test")

                                        plt.show()

                                    # Now we perform the train/test split on the dataset, returned in the shapes we require.
                                    u_train, y_train, I_test, V_test = TTS_procedure(data=data, train_len=TRAIN, test_len=TEST, warmup=WARMUP, dt=DT)
                                    end = time.monotonic()
                                    result["metrics"]["load_time"] = end - start

                                    if VERBOSE:
                                        print(f"Time to split data = {end - start} seconds")
                                        print(f"u_train shape: {u_train.shape}")
                                        print(f"y_train shape {y_train.shape}")
                                        print(f"I_test shape {I_test.shape}")
                                        print(f"V_test shape {V_test.shape}")
                                        print("\n")


                                    # ----- INITIALISE RANDOM MATRICES -----

                                    init_time_start = time.monotonic()
                                    # This step is outsourced. See "WeightsInit.py" for the full code behind the initialisation.
                                    if ORDERED:
                                        Win, Wres = initialize_sparse_orderly_weights(N, K, p, iss, rc_connectivity, seed=seed, bias=BIAS, dtype=DTYPE)
                                    elif UNIFORM:
                                        Win, Wres = initialize_uniform_weights(N, K, p, iss, in_connectivity, rc_connectivity, seed=seed, bias=BIAS)
                                    else:
                                        raise ValueError("Either UNIFORM or ORDERED must be set to true.")

                                    init_time_end = time.monotonic()
                                    result["metrics"]["init_time"] = init_time_end - init_time_start

                                    if VERBOSE:
                                        print(f"Initialising weights took {init_time_end - init_time_start} seconds.")
                                        print(f"W_in shape = {Win.shape}")
                                        print(f"W_res shape = {Wres.shape}")
                                        print("\n")


                                    # ----- BEGIN COLLECTING RESERVOIR STATES -----

                                    states_time_start = time.monotonic()
                                    # Allocate memory for the state matrix
                                    X = np.zeros(shape=(N, TRAIN-WARMUP))
                                    x = np.zeros(shape=(N, 1))

                                    for t in range(TRAIN):
                                        u = u_train[:, t].reshape(-1, 1)

                                        # Perform ESN update with the characteristic state update equation.
                                        x = (1-a)*x + a*np.tanh(np.dot(Win, u) + np.dot(Wres, x))

                                        if t >= WARMUP:  # We only store values that are generated after the warmup period.
                                            X[:, t - WARMUP] = x.ravel()

                                    states_time_end = time.monotonic()
                                    result["metrics"]["states_time"] = states_time_end - states_time_start

                                    if VERBOSE:
                                        # Printing the time elapsed to acquire reservoir states.
                                        print(f"Producing reservoir states took {states_time_end - states_time_start} seconds.")
                                        print(f"State matrix X shape: {X.shape}")

                                        # The following variable is the number of neuron you wish to plot.
                                        P = 10

                                        # Plot the activations of the reservoir neurons.
                                        plt.figure(figsize=(10, 3))
                                        plt.title(f"Activation of {P} reservoir neurons.")
                                        plt.ylabel("$reservoir activation$")
                                        plt.xlabel("$t (0.01 us)$")
                                        plt.plot(X[:P, :].T)
                                        plt.show()

                                    # ----- TRAINING THE READOUT WEIGHT MATRIX -----

                                    train_time_start = time.monotonic()
                                    # TODO: Try sparse.identity
                                    penalty = reg * np.eye(N)
                                    X_T = X.T
                                    Wout = np.dot(
                                        np.dot(y_train, X_T), np.linalg.inv(np.dot(X, X_T) + penalty)
                                    )

                                    train_time_end = time.monotonic()
                                    result["metrics"]["train_time"] = train_time_end - train_time_start

                                    if VERBOSE:
                                        print(f"W_out shape = {Wout.shape}")
                                        print(f"Solving for W_out took {train_time_end - train_time_start} seconds.")
                                        print(Wout)

                                    # ----- GENERATIVE FORECASTING -----

                                    test_time_start = time.monotonic()
                                    # Allocate memory for the forecasting results_Analysis.
                                    V_pred = np.zeros((L, TEST))

                                    # Establish the initial point to begin forecasting. We require shape (K, 1) and not (K, ), hence the reshape.
                                    u = u_train[:, -1].reshape(-1, 1)
                                    if VERBOSE:
                                        print(f"!!! u shape = {u.shape}!!!")
                                        print(f"Last training value = {u}")

                                    for t in range(TEST):
                                        # Predict the next voltage
                                        # y = np.squeeze(np.dot(Wout, x))
                                        y = np.squeeze(np.asarray(np.dot(Wout, x)))

                                        # Set the prediction matrix value at the current timestep to the current prediction.
                                        V_pred[:, t] = y

                                        u = np.array([np.squeeze(I_test[:, t]), y]).reshape(-1, 1)  # Must have shape (2, 1) at the end.

                                        # Transform the input.
                                        x = (1-a)*x + a * np.tanh(np.dot(Win, u) + np.dot(Wres, x))


                                    test_time_end = time.monotonic()
                                    result["metrics"]["test_time"] = test_time_end - test_time_start

                                    # ----- CALCULATING MODEL ERROR -----

                                    # MAE:
                                    mae = MAE(y_true=V_test.T, y_pred=V_pred.T)
                                    mse = MSE(y_true=V_test.T, y_pred=V_pred.T)
                                    rmse = RMSE(y_true=V_test.T, y_pred=V_pred.T)
                                    nrmse = NRMSE(y_true=V_test.T, y_pred=V_pred.T)
                                    r2 = R2(y_true=V_test.T, y_pred=V_pred.T)
                                    max_err = max_error(y_true=V_test.T, y_pred=V_pred.T)

                                    result["metrics"]["mae"] = mae
                                    result["metrics"]["mse"] = mse
                                    result["metrics"]["rmse"] = rmse
                                    result["metrics"]["nrmse"] = nrmse
                                    result["metrics"]["r2"] = r2
                                    result["metrics"]["max_err"] = max_err

                                    if VERBOSE:
                                        print(f"Time to forecast = {test_time_end - test_time_start} seconds.")
                                        print("\nError metrics:")
                                        print(f"Mean Absolute Error = {mae}")
                                        print(f"Mean Squared Error = {mse}")
                                        print(f"Root Mean Squared Error = {rmse}")
                                        print(f"Normalized (std) Root Mean Squared Error = {nrmse}")
                                        print(f"R-squared = {r2}")
                                        print(f"Maximum absolute error = {max_err}")

                                        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
                                        plt.subplots_adjust(hspace=0.6)

                                        # Prediction vs truth plot:
                                        axs[0].set_title("Forecasted vs True voltage")
                                        axs[0].set_xlabel("Time (ms)")

                                        if NORMALIZE:
                                            axs[0].set_ylabel("Normalized Voltage")
                                        elif STANDARDIZE1 or STANDARDIZE2:
                                            axs[0].set_ylabel("Standardized Voltage")
                                        else:
                                            axs[0].set_ylabel("Voltage (mV)")
                                        axs[0].set_ylim(-2, 6)

                                        # Plot true and forecasted voltages
                                        axs[0].plot(V_pred.T, color="r", lw=1.5, label="Forecasted")
                                        axs[0].plot(V_test.T, color="k", lw=1.0, label="True")
                                        axs[0].legend()


                                        # Plotting error over time:
                                        # Point-wise Error:
                                        errors = pointwise_error(y_true=V_test.T, y_pred=V_pred.T)
                                        axs[1].set_title(f"ESN RMSE")
                                        axs[1].set_xlabel("Time (0.01 us)")
                                        axs[1].set_ylabel("RMSE")

                                        # Plotting the pointwise error:
                                        axs[1].plot(errors, color='k')

                                        plt.show()

                                    # Add current test to overall results_Analysis for this file
                                    final_time = time.monotonic()
                                    result["metrics"]["iter_time"] = final_time - start
                                    print(f"Iter_time = {final_time - start}")
                                    results["results_Analysis"].append(result)

    rmse_values = [result["metrics"]["rmse"] for result in results["results_Analysis"]]
    order = np.argsort(rmse_values)
    results["order"] = [int(o) for o in order]

    OUTPUT_LOCATION = os.path.abspath(
        os.path.join(OUTPUT_DIR, f"{PROTOCOL}.json")
    )
    with open( OUTPUT_LOCATION, "w" ) as file:
        json.dump(results, file, indent=4)
