import scipy
import numpy as np
from math import comb, factorial
from itertools import combinations_with_replacement
from sklearn.linear_model import Ridge, Lasso


def prepare_data(signal: np.ndarray, train: int, test: int, dt: int):

    """
    Splits the signal into a train/test split for inputs and outputs.
    This case is used to predict the signal one step ahead of time.

    :param signal: The input timeseries signal, must have shape (timesteps, n-dimensions).
    :param train:
    :param test:
    :param dt:
    :return: x_train, x_test, y_train, y_test, each component has the same n-dimensions.
    """

    # Extract features and targets from the signal
    x_train = signal[0:train, :]
    y_train = signal[1:train + dt, -1].reshape(-1, 1)

    x_test = signal[train:train + test, :]
    y_test = signal[train + dt:train + test + dt, -1].reshape(-1, 1)

    return x_train, x_test, y_train, y_test


class NVAR2:

    def __init__(self, delay: int, order: int, strides: int = 1, verbose=True, bias=True):
        self.k = delay
        self.p = order
        self.s = strides
        self.verbose = verbose
        self.bias = bias
        self.features = None
        self.weights = None
        self.window = None
        self.monomial_index = None
        self.last_point = None

        if self.k < 1:
            raise ValueError("Delay (k) should be an integer greater than or equal to 1.")
        if self.s < 1:
            raise ValueError("Strides (s) should be an integer greater than or equal to 1.")
        if self.p < 1:
            raise ValueError("Order (p) should be an integer greater than or equal to 1.")

    # TODO: Produce new nonlinear fits for the dataset. Method works for doublescroll.
    def fit_polynomial(self, x: np.ndarray) -> np.ndarray:

        """
        :param x: Training data, n-dimensional timeseries data of shape (timesteps, dimensions)
        :return: An ndarray feature matrix used to train to a readout.

        """

        if type(x) is not np.ndarray:
            raise TypeError("Method input must be type np.ndarray of shape (timesteps, n_dimensions.")

        if x.ndim < 2:
            x = x.reshape(-1, 1)

        # Extracting the dimensions of the input signals.
        timesteps, n_dim = x.shape

        # Feature subset input dimensions.
        lin_dim = n_dim * self.k
        nlin_dim = comb(lin_dim + self.p - 1, self.p)  # (d + p - 1)! / (d - 1)!p!

        # Notify the user of the vector dimensions.
        if self.verbose:
            print(f"Linear dimension = {lin_dim}")
            print(f"Nonlinear dimension = {nlin_dim}")
            print(f"Total feature length = {lin_dim + nlin_dim}")

        # By default, a window for the lagged inputs is initialised as all zeros.
        window_length = 1 + (self.k - 1) * self.s
        self.window = np.zeros((window_length, n_dim), dtype="object")

        # Feature vector is made of two components, these are concatenated later.
        linear = np.zeros((timesteps, lin_dim))
        nonlinear = np.zeros((timesteps, nlin_dim))
        # nonlinear = np.zeros((timesteps, lin_dim))

        # We can pre-compute the indices of the unique entries in the p-order multiplication
        # of the linear vector. Pre-computation of monomial indices increases efficiency.
        self.monomial_index = np.array(list(combinations_with_replacement(np.arange(lin_dim), self.p)))

        for t in range(timesteps):
            # We shift all the rows of the window down one. Moving the viewed time up 1 timestep.
            self.window = np.roll(self.window, 1, axis=0)
            # Replace the first value of the window with the current value of X.
            self.window[0, :] = x[t]

            # Linear component:
            linear_features = self.window[::self.s, :].flatten()

            # Using the index list created with itertools, we perform column-wise multiplication for all
            # unique combinations of products from the linear features extracted.
            nlinear_features = np.prod(linear_features[self.monomial_index], axis=1, dtype="object")
            # nlinear_features = np.array([np.tanh(val) for val in linear_features])

            # Updating model features, variables are placed along the columns.
            linear[t, :] = linear_features
            nonlinear[t, :] = nlinear_features

        # Once each step has been processed, we combine both matrices to make one large feature matrix.
        features = np.hstack([linear, nonlinear])
        self.features = features

        return self.features

    def fit_individual(self, input_vector: np.ndarray):

        """
        For one-step-ahead prediction, we wish to fit the data one timestep at a time to form a polynomial
        representation, the output of this method is designed to be fed into self.predict.

        :param input_vector: np.ndarray of shape (1, n_dimensions).
        :return: A polynomial feature vector of shape [1,
        """

        if input_vector.ndim < 2:  # Should be shape (1, num_dimensions)
            input_vector = input_vector.reshape(-1, 1)

        # Shift every row down one, with the last row wrapping around to the end of the array.
        self.window = np.roll(self.window, 1, axis=0)
        # Replace the first row of the window with the current input values.
        self.window[0, :] = input_vector

        # From this window, we can extract the linear components, which are then used for polynomial transformation.
        linear_features = self.window[::self.s, :].flatten()
        nlinear_features = np.prod(linear_features[self.monomial_index], axis=1)
        # nlinear_features = np.array([np.tanh(val) for val in linear_features])

        # Through simple concatenation, we will produce the overall transformed feature vector.
        features = np.concatenate((linear_features.reshape(1, -1), nlinear_features.reshape(1, -1)), axis=1)

        self.last_point = input_vector[-1, -1]  # Assumes target is in the last column.

        return features

    def train_ridge_cholesky(self, features: np.ndarray, targets: np.ndarray, alpha=0.01, bias=True):

        """
        Performs ridge regression with L2 regularization between NVAR combined feature vector
        and readout targets to create a readout weight matrix. y_pred = WX. Model is trained to
        predict the input signal one timestep ahead. However, horizon can depend entirely on
        the inputs fed into the model.

        :param features: The transformed feature vector. Has size ()
        :param targets: The output vector of the desired signal the model learns to predict.
        :param ridge: The L2 regularization parameter, penalises weights based on their squared value to
                      avoid overfitting.
        :param bias: If true, a bias vector of 1 is concatenated onto the top of the training matrix.
        :return: Readout weight matrix of shape (target_dimension, feature_dimension + bias), np.ndarray.
        """

        # Perform some quick type checks to ensure proper implementation.
        if type(features) is not np.ndarray or type(targets) is not np.ndarray:
            raise TypeError("Feature and target vectors must both be type np.ndarray")

        if self.verbose:
            print(f"Training conducted via ridge regression.")

        # The warm-up phase is equal to s * k. We do not consider the features that have no delayed values associated.
        transient = self.s * self.k
        # Adjust the corresponding signal length.
        n_steps = len(features) - transient
        x_train = features[transient:]

        if self.bias:
            b = np.ones((n_steps, 1))
            # We place these biases as the first column in each row for the feature vector.
            x_train = np.hstack([b, x_train])

        # For the data that is removed in the features we must also do so with the targets.
        y = targets[transient:]
        if y.ndim < 2:
            y.reshape(-1, 1)

        # Extracting matrix dimensions.
        features_dim = x_train.shape[1]
        target_dim = y.shape[1]

        # Initialise weight matrix, begin preliminary computations:
        readout = np.zeros((target_dim, features_dim))

        YX_T = np.dot(y.T, x_train)
        XX_T = np.dot(x_train.T, x_train)
        penalty = alpha * np.identity(len(XX_T), dtype=np.float32)
        print(f"Determinant of square feature and penalty = {scipy.linalg.det(XX_T + penalty)}")

        # Using numpy
        readout[:] = np.dot(YX_T, scipy.linalg.pinvh(XX_T + penalty))
        # Using scipyscipy.linalg.pinvh(XX_T + penalty)
        # readout[:] = np.dot(YX_T, scipy.linalg.inv(XX_T + penalty))
        # readout[:] = YX_T @ scipy.linalg.pinvh(XX_T + penalty)

        self.weights = readout
        return self.weights

    def train_lasso(self, features: np.ndarray, targets: np.ndarray, alpha=0.01, bias=True):
        # Other parts of your code remain unchanged

        # Initialize Lasso model
        lasso = Lasso(alpha=alpha)

        transient = self.s * self.k
        # Adjust the corresponding signal length.
        n_steps = len(features) - transient
        x_train = features[transient:]

        if bias:
            b = np.ones((n_steps, 1))
            x_train = np.hstack([b, x_train])

        y = targets[transient:]
        if y.ndim < 2:
            y.reshape(-1, 1)

        # Fit Lasso model
        lasso.fit(x_train, y)

        self.weights = lasso.coef_.reshape(1, -1)
        return self.weights

    def train_ridge_sgd(self, features: np.ndarray, targets: np.ndarray, alpha=0.01, bias=True):
        # Other parts of your code remain unchanged

        # Initialize sgd ridge regressor.
        regressor = Ridge(alpha=alpha, solver="sag")

        transient = self.s * self.k
        # Adjust the corresponding signal length.
        n_steps = len(features) - transient
        x_train = features[transient:]

        if bias:
            b = np.ones((n_steps, 1))
            x_train = np.hstack([b, x_train])

        y = targets[transient:]
        if y.ndim < 2:
            y.reshape(-1, 1)

        # Fit Lasso model
        regressor.fit(x_train, y)

        self.weights = regressor.coef_.reshape(1, -1)
        return self.weights

    def reproduce(self, x_test) -> np.ndarray:

        """
        Recreate the training signal through simple y=WX multiplication. Ideally the model should be able to forecast
        the training data as well.

        :param x_test: The standard input vector of shape (timesteps, n_dimensions) This is transformed like
                       the signal that was trained on.
        :return: Y_pred: An array for the predicted signal of shape (timesteps, target_dimensions)
        """

        w, bias = self.weights[:, 1:], self.weights[:, :1]
        print(f"bias shape = {bias.shape}")

        # Transform the testing signal:
        x = self.fit_polynomial(x_test)

        reproduced_signal = np.dot(w, x.T) + bias

        return reproduced_signal.T

    def predict(self, x_test) -> float:

        """
        :param x_test: The standard input vector of shape (1, n_dimensions) This is transformed like
                       the signal that was trained on. Building off of the previously generated window.

        :return: Y_pred: Predicts one voltage
        """

        # Separating weight from bias.
        if self.bias:
            if self.weights.shape[1] == self.features.shape[1] + 1:
                w, bias = self.weights[:, 1:], self.weights[:, :1]
            else:
                w, bias = self.weights, np.ones((self.weights.shape[0], 1))

            # Transform the testing signal:
            x = self.fit_individual(x_test)
            # print(x.flatten())

            return (np.dot(w, x.T) + bias).T.item()

        else:
            w = self.weights

            # Transform the testing signal:
            x = self.fit_individual(x_test)
            # print(x.flatten())

            return (np.dot(w, x.T)).T.item()

    # def forecast(self, window: np.ndarray, feature_inputs: np.ndarray, timesteps: int) -> np.ndarray:
    #
    #     # TODO: Have function take in a warmup dataset and produce a window here to permit generalised prediction.
    #
    #     """
    #     Function takes in an array of length (k-1)s + 1 containing all features and outputs to generate a window of
    #     inputs to base its initial prediction on. Without it, the initial voltage prediction will be made without any
    #     influence from its previous states, resulting in a cascading error from the beginning.
    #
    #     :param window: The warmup period for the predictor, the last values are used to create the initial prediction.
    #     :param feature_inputs: The features that are kept alongside the
    #     :param timesteps:
    #     :return: An np.ndarray of the feature that is being predicted in shape (timesteps, n_dimensions)
    #     """
    #
    #     voltages = np.zeros(len(feature_inputs))
    #
    #     v = initial
    #
    #     # First we compute the new voltage value from the previously trained value.
    #     for t in range(timesteps):
    #
    #         input_array = np.array([[float(current_protocol[t]), v]])
    #
    #         v = self.predict(input_array)
    #         # print(f"Voltage = {v}")
    #         voltages[t] = v
    #         # print(v)
    #
    #     return np.array(voltages)

    def forecast(self, driving_force: np.ndarray, timesteps: int, continuation=True):

        """
        At present only permits continuation from the timeseries it has been trained on. The driving force must be
        supplied separatly

        :param known_features:
        :param timesteps:
        :param continuation:
        :return:
        """

        injected_current = np.squeeze(driving_force)
        outputs = np.zeros(shape=(2, timesteps))
        x = self.las

        for t in range(timesteps):


        if continuation:  # Rebuild the window.
            last_point = self.last_point
        else:
            last_point = 0

            window_length = 1 + (self.k - 1) * self.s
            self.window = np.zeros((window_length, n_dim), dtype="object")

        for t in range(timesteps):

