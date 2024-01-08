import numpy as np

"""
In this document we define some simple functions we can use to assess the goodness of fit in the regression task.
We provide the following metrics:
    
    MAE - Mean Absolute Error 
    MSE - Mean Squared Error 
    RMSE - Root Mean Squared Error 
    R2 - R-square

But primarily proceed with RMSE for an average error per point metric. To further assess model effectiveness, we can
split the error metrics temporally so we can assess how the predictions degrade with accumulating error. In addition, 
this allows us to see which regions of phase space have been best understood by the signal.
"""


def pointwise_error(y_true: np.ndarray, y_pred: np.ndarray, square=False) -> np.ndarray:
    if square:
        return (y_true - y_pred)**2
    else:
        return np.abs(y_true - y_pred)


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true-y_pred)**2)


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(MSE(y_true=y_true, y_pred=y_pred))

def NRMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(MSE(y_true=y_true, y_pred=y_pred)) / np.std(y_true, axis=0)[0]


def R2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.max(np.abs(y_true - y_pred))


def rolling_rmse(y_true: np.ndarray, y_pred: np.ndarray, splits: int, error_func: callable) -> list:

    """
    Calculates the RMSE of the signal across multiple windows, allowing us to see the error in model predictions
    over select regions. Also allowing us to see how the predictions degrade over time.

    :param y_true: Model predictions.
    :param y_pred: True values the model is trying to replicate.
    :param splits: The number of segments to split the signal into, default is 1. Must be an integer.
    :param error_func: The error function desired to measure the goodness of fit. Can be MAE, MSE, RMSE, R2.
    :return:
    """
    rolling_errors = []

    # Measure the length of the signals. Which of course must be of equal shape.
    n = len(y_true)

    step_size = n // splits

    for i in range(0, n, step_size):
        # Splice the current segment of the signal:
        y_true_split = y_true[i:i + step_size]
        y_pred_split = y_pred[i:i + step_size]

        # Calculate the error metric using the callable error function:
        error = error_func(y_true_split, y_pred_split)

        rolling_errors.append(error)

    return rolling_errors
