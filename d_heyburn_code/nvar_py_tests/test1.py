import os
import numpy as np
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
from NVAR1 import *

# Pulling data into programme:
cwd = os.getcwd()
data_dict = os.path.join(os.path.dirname(cwd), "data", "Normalised_Signals")
# data_dict = os.path.join(os.path.dirname(cwd), "Standardised_Signals")
print(data_dict)

# Specifying some hyperparameters.
TRAIN_SIZE = 10000
TEST_SIZE = 100000
HORIZON = 1

s = 7
k = 2

if __name__ == "__main__":

    # Importing the data.
    data_file = os.path.join(data_dict, "M_NProtocol_17")
    signal = pd.read_csv(data_file, header=0, delimiter=",", dtype="float").values

    # Optional: Different signal for testing data.
    # The point is for a signal to predict its own future, so ill-advised to measure success on predicting other
    # signals. It is however a testament to its ability to generalise.
    data_file = os.path.join(data_dict, "M_NProtocol_22")
    signal2 = pd.read_csv(data_file, header=0, delimiter=",", dtype="float").values

    # Print out dataset properties for review
    # print(f"Signal type: {type(signal)}")
    # print(f"Signal shape: {signal.shape}")

    # ----- Train/Test Split -----
    x_train = signal[0:TRAIN_SIZE, :]
    y_train = signal[1:TRAIN_SIZE+HORIZON, -1].reshape(-1, 1)

    x_test = signal2[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE, 0]
    print(x_test)
    y_test = signal2[TRAIN_SIZE+HORIZON:TRAIN_SIZE+TEST_SIZE+HORIZON, -1].reshape(-1, 1)
    # print(y_test)

    # Declaring the model.
    nvar = NVAR(delay=k, order=2, strides=s)

    # Transforming the signal.
    training_features = nvar.fit(x_train)
    # print(training_features[0])

    # Carry out regression:
    weight_matrix = nvar.train(
        features=training_features,
        targets=y_train,
        ridge=0.001,
    )
    # print(weight_matrix)
    # print(weight_matrix.shape)

    y_pred = nvar.predict(X_test=x_test)
    print(y_pred)

    # Some analysis:
    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    print(f"MAE = {mae}")
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    print(f"RMSE = {np.sqrt(mse)}")

    plt.figure(figsize=(10, 4))
    plt.plot(y_pred, color="r", lw=1.5, label="Prediction")
    plt.plot(y_test, color="k", lw=1.0, label="Ground truth")
    plt.title(f"Forecasted Membrane Voltage ({TRAIN_SIZE}/{TEST_SIZE})")
    plt.legend()

    plt.savefig("V_measured_forecast.png")
    plt.show()
