import os
import time

import numpy as np
import pandas as pd
import reservoirpy as rpy
import matplotlib.pyplot as plt

# A more optimal function for matrix inversions.
from scipy import linalg
# Importing functions used for random weight initialization.
from WeightsInit import *
# Importing functions for calculating model performance metrics.
from ErrorFunctions import *

# Configuring MatPlotLib:
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["legend.fontsize"] = 12



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


# ----- BEGIN PROGRAM -----

if __name__ == "__main__":

    # Picking which current protocol to use. Options are: 17, 21-30.
    PROTOCOL = 17

    # Set to true to enable figures and printing operations.
    VERBOSE = True

    # Temporal Parameters:
    TRAIN = 75000
    TEST = 25000
    WARMUP = 0
    DT = 1
    timesteps = np.linspace(0, 2000, TRAIN + TEST)

    # Pre-processing choice: Set both to 'False' to carry out no pre-processing.
    # We've found that for this data, the model only works if it is standardized.
    NORMALIZE = False
    STANDARDIZE1 = True
    STANDARDIZE2 = False

    # Random distribution: Only one should be true at runtime.
    ORDERED = False
    UNIFORM = True

    # Echo-State Network Parameters:
    # Input vector (K) size:
    K = 2
    # Output vector (L) size:
    L = 1
    # Number of nodes (N) inside the reservoir:
    N = 5000
    # The leaking rate (a), the fraction of the previous state preserved at each step:
    a = 0.3
    # a = 0.76
    # The spectral radius (p), amplifies or dampens the chaotic nature of the signal. <1 is typical:
    # p = 1.06
    p = 0.8
    # p = 0.98
    # Input scaling for the input vector u:
    iss = 0.5
    # iss = 0.5
    # iss = 0.62
    # Regularization for the L2 regression:
    reg = 0.0004
    # reg = 1e-4
    # Connectivity: The probability of any two nodes being connected. Including self-connections.
    in_connectivity = 1.0
    # rc_connectivity = 0.003
    rc_connectivity = 0.003
    # Miscellaneous parameters:
    SEED = 0
    np.random.seed(seed=SEED)
    DTYPE = "float32"  # Keep this as a float, but it is fine to change its size.
    BIAS = False

    start = time.perf_counter()
    # ----- IMPORTING DATASET -----
    # parent_dict = os.path.dirname(os.getcwd())
    # # In the protocol string: M=Measured, I=Integrated.
    # data_loc = os.path.join(parent_dict, "data", "Signals", f"M_Protocol_{PROTOCOL}")
    # # Leave as a DataFrame to allow for column-wise pre-processing.
    data_loc = "modelled_signal_and_error_train_p17.csv"
    data = pd.read_csv(data_loc, header=0, delimiter=',', dtype=DTYPE)[:200000:2]

    # For the integrated data:
    # data_loc = os.path.join(parent_dict, "data", "upsampled_integrated", f"I_Protocol_{PROTOCOL}")
    # # Leave as a DataFrame to allow for column-wise pre-processing.
    # data = pd.read_csv(data_loc, header=0, delimiter=',', dtype=DTYPE)[:200000:2]


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
    print(data.shape)

    # Plotting the train and test signals.
    if VERBOSE:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

        # Plotting the injected current for the training part of the signal.
        # axs[0].set_title("Injected Current")
        if NORMALIZE:
            axs[0].set_ylabel("Normalized Modelled Voltage", fontsize=10)
        elif STANDARDIZE1 or STANDARDIZE2:
            axs[0].set_ylabel("Standardized Modelled Voltage", fontsize=10)
        else:
            axs[0].set_ylabel("Modelled Voltage (mV)", fontsize=10)
        # Warmup current
        axs[0].plot(timesteps[:WARMUP], data[:WARMUP, 0], color="r", label="Warmup")
        # Training current
        axs[0].plot(timesteps[WARMUP:TRAIN], data[WARMUP:TRAIN, 0], color="k", label="Train")
        # Testing current
        axs[0].plot(timesteps[TRAIN:TRAIN + TEST], data[TRAIN:TRAIN + TEST, 0], color='b', label="Test")
        axs[0].text(0.98, 0.95, '(a)', horizontalalignment='right', verticalalignment='top', transform=axs[0].transAxes,
                    fontsize=12, fontweight='normal')

        # Plotting the Voltage signal.
        # axs[1].set_title("Membrane Voltage")
        axs[1].set_xlabel("Time (ms)", fontsize=10)
        if NORMALIZE:
            axs[1].set_ylabel("Normalized Voltage Error", fontsize=10)
        elif STANDARDIZE1 or STANDARDIZE2:
            axs[1].set_ylabel("Standardized Voltage Error", fontsize=10)
        else:
            axs[1].set_ylabel("Voltage error (mV)", fontsize=10)
        # Warmup current
        axs[1].plot(timesteps[:WARMUP], data[:WARMUP, 1], color="r", label="Warmup")
        # Training voltage
        axs[1].plot(timesteps[WARMUP:TRAIN], data[WARMUP:TRAIN, 1], color="k", label="Train")
        # Testing voltage
        axs[1].plot(timesteps[TRAIN:TRAIN + TEST], data[TRAIN:TRAIN + TEST, 1], color='b', label="Test")
        axs[1].text(0.98, 0.95, '(b)', horizontalalignment='right', verticalalignment='top', transform=axs[1].transAxes,
                    fontsize=12, fontweight='normal')

        fig.tight_layout()

        plt.show()

    # Now we perform the train/test split on the dataset, returned in the shapes we require.
    u_train, y_train, I_test, V_test = TTS_procedure(data=data, train_len=TRAIN, test_len=TEST, warmup=WARMUP, dt=DT)
    end = time.perf_counter()

    print(f"Time to split data = {end - start} seconds")
    if VERBOSE:
        print(f"u_train shape: {u_train.shape}")
        print(f"y_train shape {y_train.shape}")
        print(f"I_test shape {I_test.shape}")
        print(f"V_test shape {V_test.shape}")
        print("\n")


    # ----- INITIALISE RANDOM MATRICES -----

    init_time_start = time.monotonic()
    # This step is outsourced. See "WeightsInit.py" for the full code behind the initialisation.
    if ORDERED:
        Win, Wres = initialize_sparse_orderly_weights(N, K, p, iss, rc_connectivity, seed=SEED, bias=BIAS, dtype=DTYPE)
    elif UNIFORM:
        Win, Wres = initialize_uniform_weights(N, K, p, iss, in_connectivity, rc_connectivity, seed=SEED, bias=BIAS)
    else:
        raise ValueError("Either UNIFORM or ORDERED must be set to true.")

    init_time_end = time.monotonic()
    print(f"Initialising weights took {init_time_end - init_time_start} seconds.")

    if VERBOSE:
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

    if VERBOSE:
        # Printing the time elapsed to acquire reservoir states.
        print(f"Producing reservoir states took {states_time_end - states_time_start} seconds.")
        print(f"State matrix X shape: {X.shape}")

        # The following variable is the number of neuron you wish to plot.
        P = 10

        # Plot the activations of the reservoir neurons.
        plt.figure(figsize=(10, 3))
        # plt.title(f"Activation of {P} reservoir neurons.")
        plt.ylabel("Node Activation", fontsize=10)
        plt.xlabel("Time (ms)", fontsize=10)
        plt.plot(timesteps[WARMUP:TRAIN], X[:P, :].T, lw=0.5)
        plt.tight_layout()  # Add this line to apply tight layout
        plt.show()


    # ----- TRAINING THE READOUT WEIGHT MATRIX -----

    train_time_start = time.monotonic()
    penalty = reg * np.eye(N)
    X_T = X.T
    Wout = np.dot(
        np.dot(y_train, X_T), np.linalg.inv(np.dot(X, X_T) + penalty)
    )

    train_time_end = time.monotonic()

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
    print(f"Last training value = {u}")

    for t in range(TEST):
        # Predict the next voltage
        # y = np.squeeze(np.dot(Wout, x))
        y = np.squeeze(np.asarray(np.dot(Wout, x)))

        # Set the prediction matrix value at the current timestep to the current prediction.
        V_pred[:, t] = y

        # print(f"I_test[:, t] shape = {I_test[:, t].shape}")
        # print(f"I_test[:, t] shape = {np.squeeze(I_test[:, t].shape)}")
        # print(f"y shape = {y.shape}")
        u = np.array([np.squeeze(I_test[:, t]), y]).reshape(-1, 1)  # Must have shape (2, 1) at the end.

        # Transform the input.
        x = (1-a)*x + a * np.tanh(np.dot(Win, u) + np.dot(Wres, x))


    test_time_end = time.monotonic()
    print(f"Time to forecast = {test_time_end - test_time_start} seconds.")


    # ----- CALCULATING MODEL ERROR -----
    print("\nError metrics:")

    # MAE:
    mae = MAE(y_true=V_test.T, y_pred=V_pred.T)
    print(f"Mean Absolute Error = {mae}")

    # MSE:
    mse = MSE(y_true=V_test.T, y_pred=V_pred.T)
    print(f"Mean Squared Error = {mse}")

    # RMSE:
    rmse = RMSE(y_true=V_test.T, y_pred=V_pred.T)
    print(f"Root Mean Squared Error = {rmse}")

    # RMSE:
    nrmse = NRMSE(y_true=V_test.T, y_pred=V_pred.T)
    print(f"Normalized (std) Root Mean Squared Error = {nrmse}")

    # R-Squared
    r2 = R2(y_true=V_test.T, y_pred=V_pred.T)
    print(f"R-squared = {r2}")

    # Maximum absolute error:
    max_err = max_error(y_true=V_test.T, y_pred=V_pred.T)
    print(f"Maximum absolute error = {max_err}")

    if VERBOSE:

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        plt.subplots_adjust(hspace=-0.1)

        # Prediction vs truth plot:
        # axs[0].set_title("Forecasted vs True voltage", fontsize=10)
        # axs[0].set_xlabel("Time (ms)", fontsize=10)

        if NORMALIZE:
            axs[0].set_ylabel("Predicted Model Error", fontsize=10)
        elif STANDARDIZE1 or STANDARDIZE2:
            axs[0].set_ylabel("Predicted Model Error", fontsize=10)
        else:
            axs[0].set_ylabel("Voltage (mV)", fontsize=10)
        # axs[0].set_ylim(-2, 6)

        # Plot true and forecasted voltages
        axs[0].plot(timesteps[TRAIN:TRAIN+TEST], V_pred.T, color="r", lw=1.5, label="Forecasted")
        axs[0].plot(timesteps[TRAIN:TRAIN+TEST], V_test.T, color="k", lw=1.0, label="True")
        axs[0].legend(loc='upper right', fontsize=8)
        axs[0].text(0.02, 0.95, '(a)', horizontalalignment='left', verticalalignment='top', transform=axs[0].transAxes,
                    fontsize=12, fontweight='normal')


        # Plotting error over time:
        # Point-wise Error:
        errors = pointwise_error(y_true=V_test.T, y_pred=V_pred.T)
        # axs[1].set_title(f"ESN RMSE", fontsize=10)
        axs[1].set_xlabel("Time (ms)", fontsize=10)
        axs[1].set_ylabel("RMSE", fontsize=10)

        # Plotting the pointwise error:
        axs[1].plot(timesteps[TRAIN:TRAIN+TEST], errors, color='k')
        axs[1].text(0.02, 0.95, '(b)', horizontalalignment='left', verticalalignment='top', transform=axs[1].transAxes,
                    fontsize=12, fontweight='normal')

        fig.tight_layout()

        plt.show()
