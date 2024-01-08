import numpy as np
import scipy as sc


def identity(x):
    return x


class ESN:

    """

    A class of reservoir known as an 'Echo State Network'. Reduces a recurrent neural network down to a randomly
    initialised 'reservoir' which functions as a deterministic nonlinear transformation of a time-dependent input
    signal. Upon assimilation of a timeseries, a matrix of size (timesteps, reservoir-dim) is created; with the
    non-linearity captured by this transformation, the learning process is simplified to a linear regression task.

    In this program we implement Tikhonov (L2) ridge regression in the readout layer. Whilst this is an effective
    alternative to Recurrent NNs by removing the gradient-descent requirement, there are many hyperparameters in
    a reservoir neural network that must be considered, these are listed in the __init__ method.

    _ indicates the method is private.

    Model adapted from: "https://github.com/cknd/pyESN/blob/master/pyESN.py"

    """

    def __init__(self, input_dim: int, nodes: int, output_dim: int,
                 spectral_radius=0.99, connectivity=0.1, leak=None, noise=0.001,
                 input_scaling=None, input_shift=None,
                 feedback=False, feedback_scaling=None, feedback_shift=None,
                 readout_activation=identity, readout_activation_inverse=identity,
                 random_state=42, verbose=False):

        """
        Model arguments:

        :param input_dim: Number of input dimensions.
        :param nodes: Number of units/neurons inside the reservoir.
        :param output_dim: Number of output dimensions.
        :param spectral_radius: The absolute value of the largest eigenvalue for the recurrent weight matrix.
        :param connectivity: The fraction of neurons that are interconnected. 0 (low) - 1 (high).
        :param leak: The fraction of the previous input that is discarded in a state update.
        :param noise: The order of magnitude for the random noise applied to the signal.
        :param input_scaling: A scalar or vector of length K that multiplies each input dimension prior to observation.
        :param input_shift; A scalar or vector of length K that adds a value
        :param feedback: Dictates whether a feedback loop is created between output and reservoir.
        :param feedback_scaling: A factor applied to the output signal preceding propagation back into the reservoir.
        :param feedback_shift: An additive term applied to the target signal preceding feedback.
        :param readout_activation: The activation function for the readout layer.
        :param readout_activation_inverse: The inverse of the readout activation function.
        :param random_state: A positive integer seed, np.rand.RandomState or None.
        :param verbose: If True, produces progress updates during runtime.
        """

        # Model dimensions:
        self.K = input_dim
        self.N = nodes
        self.L = output_dim

        # Model hyperparameters:
        self.sr = spectral_radius
        self.connectivity = connectivity
        self.leak = leak
        self.noise = noise
        self.input_scaling = input_scaling
        self.input_shift = input_shift

        # Feedback mechanisms:
        self.feedback = feedback
        self.feedback_scaling = feedback_scaling
        self.feedback_shift = feedback_shift

        # Additional settings:
        self.readout_activation = readout_activation
        self.readout_activation_inverse = readout_activation_inverse
        self.random_state = random_state
        self.verbose = verbose

        # ESN accepts random_state inputs of seeds, numpy.random_state or none. At this point we settle on an answer.
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        # If not a random.RandomState object, check that the input is a valid seed.
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception(f"Invalid seed: {e}")
        else:
            raise ValueError("Invalid input for random_seed. Must be np.random.RandomState or a seed.")

        self._initialise_weights()

    @staticmethod
    def correct_dimensions(s, target_length):
        """checks the dimensionality of some numeric argument s, broadcasts it
           to the specified length if possible.

        Args:
            s: None, scalar or 1D array
            target_length: expected length of s

        Returns:
            None if s is None, else numpy vector of length target_length
        """
        if s is not None:
            s = np.array(s)
            if s.ndim == 0:
                s = np.array([s] * target_length)
            elif s.ndim == 1:
                if not len(s) == target_length:
                    raise ValueError("arg must have length " + str(target_length))
            else:
                raise ValueError("Invalid argument")
        return s

    def _initialise_weights(self):

        # First set up the reservoir weight matrix where all values are centered around zero.
        # All values are sampled from a uniform distribution [0, 1] and are rescaled to [-0.5, 0.5]
        # The matrix is also rescaled such that the spectral radius matches the user-specified value.
        W_res = self.random_state_.rand(self.N, self.N) - 0.5
        W_res[self.random_state_.rand(*W_res.shape) > self.connectivity] = 0
        sr_init = np.max(np.abs(np.linalg.eigvals(W_res)))
        self.W_res = W_res * (self.sr / sr_init)

        # Input weights are also sampled from a uniform distribution.This has range [-1, 1]
        self.W_in = (self.random_state_.rand(self.N, self.K) * 2) - 1
        # Feedback weights, initialised the same way as the input weights.
        self.W_fb = (self.random_state_.rand(self.N, self.L) * 2) - 1

    def _update(self, state, input_signal, output_signal):

        if self.feedback:
            linear_state = (self.W_res @ state) + (self.W_in @ input_signal) + (self.W_fb @ output_signal)
        else:
            linear_state = (self.W_res @ state) + (self.W_in @ input_signal)

        nonlinear_state = self.leak * np.tanh(linear_state) + self.noise * (self.random_state_.rand(self.N) - 0.5)

        return nonlinear_state

    def _scale_inputs(self, inputs):
        """
        For each entry, k, in the K-dimensional input vector, it is multiplied by the kth entry in the input-scaling
        argument, then adds the kth entry from the input_shift argument. If either argument is a scalar then it is
        broadly applied to all input dimensions.

        :param inputs: The input vector preceding the reservoir.
        :return: The rescaled signal.
        """

        # TODO: See if this can be shrunk to "if self.input_scaling:"
        if self.input_scaling is not None:
            inputs = inputs @ np.diag(self.input_scaling)
        if self.input_shift is not None:
            inputs = inputs + self.input_shift

        return inputs

    def _scale_outputs(self, outputs):
        """
        Applies a scale factor and a shift to the output signals. Unlike the input scaling, the action is not dependent
        on the element's position. All scales are applied broadly.

        :param outputs: The output vector post-readout.
        :return: Rescaled output vector
        """

        if self.feedback_scaling is not None:
            outputs = outputs * self.feedback_scaling
        if self.feedback_shift is not None:
            outputs = outputs + self.feedback_shift

        return outputs

    def _unscale_outputs(self, scaled_outputs):
        """
        Reverses the operations applied by scale outputs

        :param scaled_outputs: The scaled output vector.
        :return: Output vector without scaling.
        """

        if self.feedback_shift is not None:
            scaled_outputs = scaled_outputs - self.feedback_shift
        if self.feedback_scaling is not None:
            scaled_outputs = scaled_outputs / self.feedback_scaling

        return scaled_outputs

    def fit(self, inputs: np.ndarray, outputs: np.ndarray, inspect=False):

        """
        Propagates signal through the reservoir, performing the nonlinear transformation of the signal and
        subsequently training the readout weights.

        :param inputs: Array of size (time-steps, K)
        :param outputs: Array fo size (time-steps, L)
        :param inspect: If true, produces a visualization of the reservoir states.

        :return: The network's output on the training data after regression. Should approximately equal outputs.
        """

        # Rescale the vectors if necessary:
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        outputs_scaled = self._scale_outputs(outputs)

        # Propagate signal through the reservoir.
        if self.verbose:
            print("Collecting reservoir states.")
        states = np.zeros((inputs.shape[0], self.N))
        for t in range(1, inputs.shape[0]):
            states[t, :] = self._update(states[t-1], inputs_scaled[t, :], outputs_scaled[t-1, :])

        # Perform regression on the signal:
        if self.verbose:
            print("Performing regression on reservoir states.")
        # TODO: Implement a better method for removing the transient. Could be yet another hyperparameter.
        transient = min(int(inputs.shape[1] / 10), 100)
        # We extend the feature matrix to include a bias, and the raw signals.
        bias = np.zeros(shape=(inputs.shape[0], 1))
        features = np.concatenate((inputs_scaled, states), axis=1)

        # Solve for W_out:
        # TODO: Would be improved by real ridge regression; although noise injection can be adequate.
        inverse_matrix = np.linalg.pinv(features[transient:, :])

        self.W_out = (inverse_matrix @ self.readout_activation_inverse(outputs_scaled[transient:, :])).T

        # Have the model remember the previous states in case prediction begins immediately after.
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = outputs_scaled[-1, :]

        # optionally visualize the collected states
        if inspect:
            from matplotlib import pyplot as plt
            # (^-- we depend on matplotlib only if this option is used)
            plt.figure(figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(features.T, aspect='auto', interpolation='nearest')
            plt.colorbar()

    def predict(self, inputs, continuation=True):
        """
        Apply the learned weights to the network's responses to new inputs.

        :param inputs:
        :param continuation:
        :return:
        """

        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        timesteps = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.N)
            lastinput = np.zeros(self.K)
            lastoutput = np.zeros(self.L)

        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        states = np.vstack([laststate, np.zeros((timesteps, self.N))])
        outputs = np.vstack([lastoutput, np.zeros((timesteps, self.L))])

        for t in range(timesteps):
            states[t + 1, :] = self._update(states[t, :], inputs[t + 1, :], outputs[t, :])
            outputs[t + 1, :] = self.readout_activation(np.dot(self.W_out, np.concatenate([states[t + 1, :], inputs[t + 1, :]])))

        return self._unscale_outputs(self.readout_activation(outputs[1:]))