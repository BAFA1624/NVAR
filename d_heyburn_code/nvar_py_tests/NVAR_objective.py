from NVAR1 import *
from time import monotonic
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_data(signal: np.ndarray, train: int, test: int, dt: int):
    """
    Splits the signal into a train/test split for inputs and outputs.
    This case is used to predict the signal one step ahead of time.

    :param signal: The input timeseries signal, must have shape (timesteps, n-dimensions).
    :return: x_train, x_test, y_train, y_test, each component has the same n-dimensions.
    """

    # Extract features and targets from the signal
    x_train = signal[0:train, :]
    y_train = signal[1:train + dt, -1].reshape(-1, 1)

    x_test = signal[train:train + test, :]
    y_test = signal[train + dt:train + test + dt, -1].reshape(-1, 1)

    return x_train, x_test, y_train, y_test

# TODO: Learn from one good signal, and test on 5 instead of training on 5 and testing separately.
def objective(params: dict):

    # Setting the parameters for a given search.
    s = params["k"]
    k = params["s"]
    r = params["r"]
    p = params['p']
    signals = params["signals"]
    TRAIN = params["train"]
    TEST = params["test"]
    dt = params["dt"]

    rmse_list = []
    mae_list = []
    r2_list = []
    time_list = []

    # Changing signal at each iteration, to generalise the learning of unique current protocols.
    for signal in signals:

        # Splitting data:
        x_train, x_test, y_train, y_test = prepare_data(signal, train=TRAIN, test=TEST, dt=dt)

        start_time = monotonic()

        # Initialise NVAR:
        nvar = NVAR(delay=k, strides=s, order=p)

        # Transform training signal.
        features = nvar.fit(x_train)

        # Perform regression.
        weights = nvar.train(features=features, targets=y_train, ridge=r)

        # Make predictions.
        y_pred = nvar.predict(X_test=x_test)

        end_time = monotonic()
        time_elapsed = end_time - start_time

        # Calculate loss metrics:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        time_list.append(time_elapsed)

    # Average these scores
    average_rmse = np.mean(rmse_list)
    average_mae = np.mean(mae_list)
    average_r2 = np.mean(r2_list)
    average_time = np.mean(time_list)

    return {'loss': average_rmse, 'status': 'ok', 'mae': average_mae, 'r2': average_r2, 'time': average_time,
            'delay': k, 'stride': s, 'ridge': r, "train": TRAIN}
