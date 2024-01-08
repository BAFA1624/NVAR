import os
import pandas as pd
import matplotlib.pyplot as plt

from NVAR2 import *


if __name__ == "__main__":

    poly_orders = [1, 2, 3, 4, 5]
    delays = [1, 2, 3, 4, 5]
    alpha = 1e-5

    # Set up dimensions for the plots

    fig, axs = plt.subplots(
        ncols=len(delays),
        nrows=len(poly_orders),
        figsize=(10, 10)
    )
    fig.suptitle(f"Standardised Membrane Voltage; x-axis = timesteps; penalty = {alpha}")
    fig.tight_layout(pad=1.3)

    # Import the data:
    cwd = os.getcwd()
    data_dict = os.path.join(os.path.dirname(cwd), "data", "Normalised_Signals")
    data_dict1 = os.path.join(os.path.dirname(cwd), "data", "signals")
    data_file = os.path.join(data_dict, "M_NProtocol_21")
    signal = 1 * pd.read_csv(data_file, header=0, delimiter=",", dtype="float").values

    # Prepare the data:
    TRAIN = 15000
    TEST = 15000
    dt = 1

    # Train
    x_train = signal[0:TRAIN, :]
    repeat_test = signal[0:TRAIN, 0]
    y_train = signal[dt:TRAIN+dt, -1].reshape(-1, 1)
    y_init1 = float(signal[0, -1])
    # Test
    x_test = signal[TRAIN:TRAIN+TEST, 0]
    y_test = signal[TRAIN+dt:TRAIN + TEST + dt, -1].reshape(-1, 1)
    y_init2 = float(signal[TRAIN, -1])

    # Importing the data.
    data_file = os.path.join(data_dict, "M_NProtocol_17")
    signal = pd.read_csv(data_file, header=0, delimiter=",", dtype="float").values

    for i, p in enumerate(poly_orders):
        for j, k in enumerate(delays):

            # Initialise the model:
            nvar = NVAR2(delay=k, strides=2, order=p, bias=True)
            # Transform feature vector
            training_features = nvar.fit_polynomial(x_train)

            # Train model
            weights = nvar.train_ridge_sgd(
                features=training_features,
                targets=y_train,
                alpha=alpha,
                bias=True
            )
            print(weights)

            # Make predictions on test data:
            y_pred = nvar.forecast(initial=y_init1, current_protocol=repeat_test, timesteps=TEST)
            # print(y_pred[:5000].flatten())

            # Plot results_Analysis
            axs[i, j].set_title(f"(k={k}, s={2}, p={p})")
            axs[i, j].plot(1e0 * y_pred, color="r", lw=1.5, label="Forecast")
            axs[i, j].plot(y_test, color="k", lw=1.0, label="Ground truth")

    plt.show()
