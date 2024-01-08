import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge, ESN
from reservoirpy.observables import rmse, rsquare


def optimizer2D(dataset: np.ndarray,
                config: str,  # Takes a JSON filepath (or equivalent) as a string input.
                *,
                train_interval: int,
                test_interval: int,
                horizon: int,
                N: int,
                sr: float,
                lr: float,
                iss: float,
                ridge: float,
                seed: int):
    """

    :param horizon:
    :param train_interval:
    :param test_interval:
    :param dataset: A list of 2D tuples of form ( (X_train, y_train) , (X_test, y_test) )
    :param config: A configuration file detailing the hyperparameter domains to explore.
    :param N: Number of units in the reservoir neural network.
    :param sr: The largest absolute eigenvalue of the reservoir weight matrix.
    :param lr: Controls the time constant of the ESN, adjusting its short-term memory.
    :param iss: A coefficient applied to the input weight matrix, adding a gain to the reservoir inputs.
    :param ridge: L2 regularization parameter for the readout regressor.
    :param seed:  The random-number generator seed. Held constant in experiments for reproducibility. But varies
                  between instances to circumvent initial biases.


    :return:
    """

    # We set the number of random ESNs that will be trialled with each set of parameters.
    num_instances = config["instances_per_trial"]

    system_seed = seed

    # Establishing our choice of metrics we'd like to measure. Loss is NRMSE.
    r2s = []
    losses = []

    for n in range(num_instances):

        # ----- DATASET -----
        train_size = train_interval * (num_instances + 1)
        test_size = test_interval

        x_train = dataset[0:train_size, :]
        y_train = dataset[1:train_size + horizon, -1].reshape(-1, 1)

        x_test = dataset[train_size:train_size + test_size, :]
        y_test = dataset[train_size + horizon:train_size + test_size + horizon, -1].reshape(-1, 1)

        # ----- RESERVOIR -----
        reservoir = Reservoir(units=N,
                              sr=sr,
                              lr=lr,
                              input_scaling=iss,
                              seed=system_seed
                              )

        # ----- REGRESSOR ------`
        readout = Ridge(ridge=ridge)

        # ----- ECHO STATE NETWORK -----
        # TODO: Re-run experiment later w/ feedback enabled.
        model = ESN(reservoir=reservoir,
                    readout=readout,
                    workers=-1,
                    backend="threading"
                    )

        # Train model:
        y_pred = model.fit(x_train, y_train).run(x_test)

        # Perform error calculations:
        loss = rmse(y_test, y_pred)
        r2 = rsquare(y_test, y_pred)

        # Append these values to their respective lists:
        losses.append(loss)
        r2s.append(r2)

        # Change the variable seed between instances. (Same range across all unique instances)
        system_seed += 1

    # Return a dict of the error metrics respective means for a set of instances.
    return {"loss": np.mean(losses),
            "r2": np.mean(r2s)}
