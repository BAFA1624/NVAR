import os
import pandas as pd
import matplotlib.pyplot as plt

from NVAR2 import *

if __name__ == "__main__":

    cwd = os.getcwd()
    data_dict = os.path.join(os.path.dirname(cwd), "data", "Normalised_Signals")
    data_file = os.path.join(data_dict, "M_NProtocol_21")
    signal = pd.read_csv(data_file, header=0, delimiter=",", dtype="float").values

    # If signal is in the same directory, comment out above and use the following command:
    # signal_dict = os.path.join(os.getcwd(), "M_NProtocol_21")
    # signal = pd.read_csv(signal_dict, header=0, delimiter=",", dtype="float").values

    # ReservoirPy was doing phenomenally with an observation window of ~15000.
    TRAIN = 15000
    TEST = 100000
    HORIZON = 1
    ksp = (5, 2, 3)  # This is for delay, stride and polynomial order respectively.

    # Training data:
    x_train = signal[0:TRAIN, :]
    # Train on the next step.
    y_train = signal[1:TRAIN + HORIZON, -1].reshape(-1, 1)
    # Or train on the difference between the current step and the next.
    dx_train = (signal[1:TRAIN + HORIZON, -1] - signal[0:TRAIN, -1]).reshape(-1, 1)

    # Actual test
    x_test = signal[TRAIN:TRAIN + TEST, :-1]  # All columns except for the last.
    y_init = signal[ksp[0]*ksp[1], -1]  # The first item of the last column.
    y_test = signal[TRAIN + HORIZON:TRAIN + TEST + HORIZON, -1].reshape(-1, 1)

    # Trying to recreate the results_Analysis:
    # x_test = signal[0:TRAIN, 0].tolist()
    # print(x_test[0])
    # y_init = signal[ksp[0]*ksp[1], -1]  # We start prediction after a transient period.
    # print(type(y_init))
    # y_test = signal[HORIZON:TRAIN + HORIZON, -1].reshape(-1, 1)

    # ----- Declare the model -----
    nvar = NVAR2(delay=ksp[0], strides=ksp[1], order=ksp[2])
    training_features = nvar.fit_polynomial(x_train)

    # print(nvar.monomial_index)

    # Ridge regression: Raise alpha to ~1e6 for predictions not to explode. 0.001 to reproduce signal.
    weights = nvar.train_ridge_sgd(
        features=training_features,
        targets=y_train,
        alpha=0.001,  # This value so far needs to be ridiculously high for the programme to work at all.
        bias=True
    )
    print(weights)

    # Reproduces the signal, appears to work well but there is a strange offset.
    y_reproduced = nvar.reproduce(x_train)
    plt.figure(figsize=(10, 4))
    plt.plot(1e0 * (y_reproduced), color="r", lw=1.5, label="Prediction")
    plt.plot(y_train, color="k", lw=1.0, label="Ground truth")
    plt.title(f"Forecasted Membrane Voltage [({TRAIN}/{TEST}), (k={ksp[0]}, s={ksp[1]}, p={ksp[2]}]")
    plt.legend()
    plt.show()

    # weights = nvar.train_lasso(
    #     features=training_features,
    #     targets=y_train,
    #     alpha=1e-4,  # This value needs to sit at this order of magnitude not to break the prediction.
    #     bias=False
    # )

    y_pred = nvar.forecast(initial=float(y_init), current_protocol=x_test, timesteps=TEST)
    # print(len(y_pred))

    plt.figure(figsize=(10, 4))
    # If you've used ridge, there is no need to change the scale factor, for lasso, you may need to scale up the values.
    plt.plot(1e0*(y_pred - 1), color="r", lw=1.5, label="Prediction")
    plt.plot(y_test, color="k", lw=1.0, label="Ground truth")
    plt.title(f"Forecasted Membrane Voltage [({TRAIN}/{TEST}), (k={ksp[0]}, s={ksp[1]}, p={ksp[2]}]")
    plt.legend()
    plt.show()
