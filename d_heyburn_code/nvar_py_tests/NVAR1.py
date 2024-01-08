import numpy as np
import pandas as pd
from math import comb
from itertools import combinations_with_replacement


class NVAR:

    def __init__(self, delay: int, order: int, strides: int = 1):
        self.k = delay
        self.p = order
        self.s = strides
        self.linear = None
        self.nonlinear = None
        self.features = None
        self.readout = None
        self.window = None

        if self.k < 1:
            raise ValueError("Delay (k) should be an integer greater than or equal to 1.")
        if self.s < 1:
            raise ValueError("Strides (s) should be an integer greater than or equal to 1.")
        if self.p < 1:
            raise ValueError("Order (p) should be an integer greater than or equal to 1.")

    def fit(self, X: np.ndarray) -> np.ndarray:

        """
        :param X: Training data, n-dimensional timeseries data of shape (timesteps, dimensions)
        :return: An ndarray feature matrix used to train to a readout.
        """

        if type(X) is not np.ndarray:
            raise TypeError("Method input must be type np.ndarray of shape (timesteps, n_dimensions.")

        if X.ndim < 2:
            X = X.reshape(-1, 1)

        # Extracting the dimensions of the input signals.
        timesteps, n_dim = X.shape

        # Feature subset input dimensions.
        lin_dim = n_dim * self.k
        nlin_dim = comb(lin_dim + self.p - 1, self.p)  # (d + p - 1)! / (d - 1)!p!

        # By default, a window for the lagged inputs is initialised as all zeros.
        window_length = 1 + (self.k - 1) * self.s
        window = np.zeros((window_length, n_dim))

        # Feature vector is made of two components, these are concatenated later.
        self.linear = np.zeros((timesteps, lin_dim))
        self.nonlinear = np.zeros((timesteps, nlin_dim))

        # We can pre-compute the indices of the unique entries in the p-order multiplication
        # of the linear vector. Pre-computation of monomial indices increases efficiency.
        monomial_index = np.array(list(combinations_with_replacement(np.arange(lin_dim), self.p)))

        for t in range(timesteps):
            # We shift all the rows of the window up one. Moving the viewed time up 1 timestep.
            window = np.roll(window, -1, axis=0)
            # Replace the last value of the window with the current value of X.
            window[-1, :] = X[t]

            # Linear component:
            linear_features = window[::self.s, :].flatten()

            # Using the index list created with itertools, we perform column-wise multiplication for all
            # unique combinations of products from the linear features extracted.
            nlinear_features = np.prod(linear_features[monomial_index], axis=1)

            # Updating model features, variables are placed along the columns.
            self.linear[t, :] = linear_features
            self.nonlinear[t, :] = nlinear_features

        # Once each step has been processed, we combine both matrices to make one large feature matrix.
        # feature = np.hstack((self.linear, self.nonlinear))
        features = np.c_[self.linear, self.nonlinear]
        self.features = features

        return self.features

    def train(self,
              features: np.ndarray,
              targets: np.ndarray,
              ridge=0.01,
              bias=True):

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

        if type(features) is not np.ndarray or type(targets) is not np.ndarray:
            raise TypeError("Feature and target vectors must both be type np.ndarray")

        # The warm-up phase is equal to s * k.
        transient = self.s * self.k
        # Remove the transient from the signal.
        n_steps = len(features) - transient
        X_train = features[transient:]

        if bias:
            b = np.ones((n_steps, 1))
            # We place these biases as the first column in each row for the feature vector.
            X_train = np.c_[b, X_train]

        # For the data that is removed in the features we must also do so with the targets.
        Y = targets[transient:]
        if Y.ndim < 2:
            Y.reshape(-1, 1)

        # Extracting matrix dimensions.
        features_dim = X_train.shape[1]
        target_dim = Y.shape[1]

        # Initialise weight matrix, begin preliminary computations:
        readout = np.zeros((target_dim, features_dim))

        YX_T = np.dot(Y.T, X_train)
        XX_T = np.dot(X_train.T, X_train)
        penalty = ridge * np.identity(len(XX_T), dtype=np.float64)

        readout[:] = np.dot(YX_T, np.linalg.inv(XX_T + penalty))
        self.readout = readout

        return readout

    def predict(self, X_test) -> np.ndarray:

        """
        :param Wout: The weight matrix produced in training. Shape (target_dimension, features + bias)
        :param X_test: The standard input vector of shape (timesteps, n_dimensions) This is transformed like
                       the signal that was trained on.
        :return: Y_pred: An array for the predicted signal of shape (timesteps, target_dimensions)
        """

        W, bias = self.readout[:, 1:], self.readout[:, :1]
        print(f"bias shape = {bias.shape}")

        # Transform the testing signal:
        X = self.fit(X_test)

        predicted_signal = np.dot(W, X.T) + bias

        return predicted_signal.T



