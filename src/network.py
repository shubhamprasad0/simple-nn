import numpy as np
from utils import activation
from utils import activation_gradient


class network:
    ''' This class defines the structure and functions of a neural network. '''

    def __init__(self, sizes, X, Y):
        ''' This constructor initializes the important aspects of
        a neural network.

        Arguments:
        sizes: a list of sizes of each layer of the neural network
               (including the input layer)
        '''
        self.L = len(sizes)  # no. of layers including the input layer.
        self.W = {}  # dictionary to store the weight parameters of each layer.

        # dictionary to store partial derivatives of weights with respect to
        # the cost function.
        self.dW = {}

        self.b = {}  # dictionary to store bias parameters of each layer.

        # dictionary to store partial derivatives of bias with respect to the
        # cost function.
        self.db = {}

        self.A = {}  # dictionary to store the activations of each layer.

        # dictionary to store the partial derivatives of activations with
        # respect to the cost function.
        self.dA = {}

        self.Z = {}  # dictionary to store the weighted inputs of each layer.

        # dictionary to store the partial derivatives of weighted inputs with
        # respect to the cost function.
        self.dZ = {}

        self.Y = Y  # numpy array to store the actual labels.
        self.A[0] = X  # A[0] is the input feature
        self.m = X.shape[1]
        self.sizes = sizes
        self.initialize_parameters()
        self.cost = 0

    def initialize_parameters(self):
        ''' This method initializes the parameters of the neural network.
        The 'W' parameters are initialized randomly, and 'b' parameters are
        initialized with zeros.

        The dimensions are as follows:
        W['l'] : sizes[l] * sizes[l - 1]
        b['l'] : sizes[l] * 1
        '''

        for l in range(self.L - 1):
            self.W[l + 1] = np.random.rand(self.sizes[l + 1],
                                           self.sizes[l]) * 0.01
            self.b[l + 1] = np.zeros((self.sizes[l + 1], 1))

    def forward_propagation(self):
        ''' This method performs the forward propagation step of the neural
        network.
        '''

        L = self.L

        for l in range(L - 2):
            self.Z[l + 1] = np.dot(self.W[l + 1], self.A[l]) + self.b[l + 1]
            self.A[l + 1] = activation(self.Z[l + 1], 'relu')

        self.Z[L - 1] = np.dot(self.W[L - 1], self.A[L - 2]) + self.b[L - 1]
        self.A[L - 1] = activation(self.Z[L - 1], 'sigmoid')

    def compute_cost(self):
        ''' This method computes the cross-entropy cost. '''

        L = self.L
        m = self.m
        Y = self.Y
        Y_hat = self.A[L - 1]

        self.cost = -(np.sum(np.sum(Y * np.log(Y_hat))) +
                      np.sum(np.sum((1 - Y) * np.log(1 - Y_hat)))) / m
        return self.cost

    def backward_propagation(self):
        ''' This method performs the backward propagation to find the
        partial derivatives required for gradient descent or any other
        optimization algorithm.
        '''

        L = self.L - 1
        self.dZ[L] = self.A[L] - self.Y
        self.dW[L] = np.dot(self.dZ[L], self.A[L - 1].T) / self.m
        self.db[L] = np.sum(self.dZ[L], axis=1, keepdims=True) / self.m

        for l in reversed(range(L)[1:]):
            self.dZ[l] = np.dot(self.W[l + 1].T, self.dZ[l + 1]) * \
                activation_gradient(self.Z[l], 'relu')
            self.dW[l] = np.dot(self.dZ[l], self.A[l - 1].T) / self.m
            self.db[l] = np.sum(self.dZ[l], axis=1, keepdims=True) / self.m

    def update_parameters(self, learning_rate):
        ''' This method performs the gradient descent update to
        minimize the cost function and find out the optimal values of
        the parameters W and b.

        Arguments:
        learning_rate : controls the rate which with the values of the
                        parameters change.
        '''

        for l in range(self.L)[1:]:
            self.W[l] -= learning_rate * self.dW[l]
            self.b[l] -= learning_rate * self.db[l]

    def nn_model(self, learning_rate=0.01, num_iter=100):
        ''' This method performs the gradient descent algorithm to
        minimize the cost function and find out the optimal values of the
        parameters W and b.

        Arguments:
        learning_rate : controls the rate which with the values of the
                        parameters change.
        num_iter : number of iterations of gradient descent to perform.
        '''

        for i in range(num_iter):
            self.forward_propagation()
            self.compute_cost()
            self.backward_propagation()
            self.update_parameters(learning_rate)

            print('Iteration', i, ':', self.cost)

    def predict(self, X):
        ''' This method predicts the label for a given image X.

        Arguments:
        X : the given image vector for which prediction has to be done.

        Returns:
        Y_hat : the predicted label for the input X.
        '''

        L = self.L - 1
        self.A[0] = X

        for l in range(L - 1):
            self.Z[l + 1] = np.dot(self.W[l + 1], self.A[l]) + self.b[l + 1]
            self.A[l + 1] = activation(self.Z[l + 1], 'relu')

        self.Z[L] = np.dot(self.W[L], self.A[L - 1]) + self.b[L]
        self.A[L] = activation(self.Z[L], 'sigmoid')

        Y_hat = self.A[L]
        return Y_hat
