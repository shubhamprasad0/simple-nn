import random
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
        X : training input features
        Y : labels of training data
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

        # number of training examples
        self.m = X.shape[1]

        self.sizes = sizes

        self.cost = 0
        self.mini_batch_size = 0

        # list to store the mini-batches
        self.mini_batches = []

        self.V_dW = {}
        self.V_db = {}
        self.S_dW = {}
        self.S_db = {}
        self.V_dW_corr = {}
        self.V_db_corr = {}
        self.S_dW_corr = {}
        self.S_db_corr = {}
        self.lmbda = 0

        # initialize the parameters of the network
        self.initialize_parameters()

    def create_mini_batches(self):
        ''' This method creates mini-batches of training examples for
        mini-batch gradient descent

        Returns:
        mini_batches : a list of all the mini-batches
        '''

        mini_batch_size = self.mini_batch_size
        m = self.m
        num_mini_batch = m // mini_batch_size

        mini_batches = []
        X = self.A[0]
        Y = self.Y

        # randomly shuffle the training set
        np.random.seed(random.randint(0, self.m))
        random_order = np.random.permutation(m)
        X = X[:, random_order]
        Y = Y[:, random_order]

        # create mini-batches
        for i in range(num_mini_batch):
            mini_batches.append((X[:, i *
                                   mini_batch_size: (i +
                                                     1) *
                                   mini_batch_size], Y[:, i *
                                                       mini_batch_size: (i +
                                                                         1) *
                                                       mini_batch_size]))

        # final mini-batch of size (possible zero) less than mini_batch_size
        if m % mini_batch_size != 0:
            mini_batches.apppend(
                (X[:, num_mini_batch * mini_batch_size:],
                 Y[:, num_mini_batch * mini_batch_size:]))

        return mini_batches

    def initialize_parameters(self):
        ''' This method initializes the parameters of the neural network.
        The 'W' parameters are initialized randomly, and 'b' parameters are
        initialized with zeros.

        The dimensions are as follows:
        W['l'] : sizes[l] * sizes[l - 1]
        b['l'] : sizes[l] * 1
        '''

        for l in range(self.L)[1:]:
            self.W[l] = np.random.rand(self.sizes[l],
                                       self.sizes[l - 1]) * 0.01
            self.b[l] = np.zeros((self.sizes[l], 1))
            self.V_dW[l] = np.zeros((self.sizes[l], self.sizes[l - 1]))
            self.V_db[l] = np.zeros((self.sizes[l], 1))
            self.V_dW_corr[l] = np.zeros((self.sizes[l], self.sizes[l - 1]))
            self.V_db_corr[l] = np.zeros((self.sizes[l], 1))
            self.S_dW[l] = np.zeros((self.sizes[l], self.sizes[l - 1]))
            self.S_db[l] = np.zeros((self.sizes[l], 1))
            self.S_dW_corr[l] = np.zeros((self.sizes[l], self.sizes[l - 1]))
            self.S_db_corr[l] = np.zeros((self.sizes[l], 1))

    def forward_propagation(self, X, Y, output_activation='sigmoid'):
        ''' This method performs the forward propagation step of the neural
        network.

        Arguments:
        output_activation: activation function applied to the output layer

                           'sigmoid': apply sigmoid activation to the output
                                      layer with cross-entropy cost function.
                           'softmax': apply softmax activation to the output
                                      layer with log-likelihood cost function.

        Returns:
        forward_cache : dictionary containing values A and Z for the current
                        mini-batch

        '''

        L = self.L - 1
        A = self.A.copy()
        A[0] = X
        Z = {}
        W = self.W
        b = self.b

        # catch to pass intermediate results to backprop step
        forward_cache = {}

        for l in range(L - 1):
            Z[l + 1] = np.dot(W[l + 1], A[l]) + b[l + 1]
            A[l + 1] = activation(Z[l + 1], 'relu')

        Z[L] = np.dot(W[L], A[L - 1]) + b[L]
        A[L] = activation(Z[L], output_activation)

        # store intermediate results in cache
        forward_cache['A'] = A
        forward_cache['Z'] = Z
        forward_cache['Y'] = Y

        return forward_cache

    def compute_cost(self, forward_cache, lmbda, output_activation='sigmoid'):
        ''' This method computes the cost function.

        Arguments:
        output_activation: determines the activation function applied to the
                           output layer, based on which we select the
                           appropriate cost function.

                           options :-

                           'sigmoid': sigmoid activation function applied to the
                                      output layer. This means that the cross-
                                      entropy cost function has to be used.

                           'softmax': softmax activation function applied to the
                                      output layer. This means that the log-
                                      likelihood cost function has to be used.
        Returns:
        cost : the computed cost function
        '''

        L = self.L - 1
        A = forward_cache['A']
        m = A[0].shape[1]
        Y = forward_cache['Y']
        Y_hat = A[L]
        lmbda = self.lmbda

        if output_activation == 'sigmoid':
            # cross-entropy cost function
            self.cost = -(np.sum(np.sum(Y * np.log(Y_hat))) +
                          np.sum(np.sum((1 - Y) * np.log(1 - Y_hat)))) / m
        elif output_activation == 'softmax':
            # log-likelihood cost function
            self.cost = -(np.sum(np.sum(Y * np.log(Y_hat)))) / m

        regularization_term = 0
        for l in range(L)[1:]:
            regularization_term += np.sum(np.sum(self.W[l] ** 2))
        regularization_term /= (2 * self.m)
        regularization_term *= lmbda
        self.cost += regularization_term

        return self.cost

    def backward_propagation(self, lmbda, forward_cache):
        ''' This method performs the backward propagation to find the
        partial derivatives required for gradient descent or any other
        optimization algorithm.
        '''

        A = forward_cache['A']
        Z = forward_cache['Z']
        Y = forward_cache['Y']
        m = A[0].shape[1]
        lmbda = self.lmbda

        L = self.L - 1  # self.L includes the input layer as well
        self.dZ[L] = A[L] - Y
        self.dW[L] = np.dot(self.dZ[L], A[L - 1].T) / m + \
            (lmbda / self.m) * self.W[L]
        self.db[L] = np.sum(self.dZ[L], axis=1, keepdims=True) / m

        for l in reversed(range(L)[1:]):
            self.dZ[l] = np.dot(self.W[l + 1].T, self.dZ[l + 1]) * \
                activation_gradient(Z[l], 'relu')
            self.dW[l] = np.dot(self.dZ[l], A[l - 1].T) / \
                m + (lmbda / self.m) * self.W[l]
            self.db[l] = np.sum(self.dZ[l], axis=1, keepdims=True) / m

    def update_parameters(self, learning_rate, optimizer, num_epoch, **kwargs):
        ''' This method performs the gradient descent update to
        minimize the cost function and find out the optimal values of
        the parameters W and b.

        Arguments:
        learning_rate : controls the rate which with the values of the
                        parameters change.
        optimizer : the optimization algorithm used to find parameters.

                    options :-

                    'gradient_descent' : usual gradient descent optimizer
                    'momentum' : gradient descent with momentum
                    'rms_prop' : gradient descent with RMSprop optimizer
                    'adam' : adam optimizer
        '''

        if optimizer == 'gradient_descent':
            for l in range(self.L)[1:]:
                self.W[l] -= learning_rate * self.dW[l]
                self.b[l] -= learning_rate * self.db[l]
        elif optimizer == 'momentum':
            # if beta=0, it is same as simple gradient descent without momentum
            beta1 = kwargs.get('beta1', 0)
            for l in range(self.L)[1:]:
                self.V_dW[l] = beta1 * \
                    self.V_dW[l] + (1 - beta1) * self.dW[l]
                self.V_db[l] = beta1 * \
                    self.V_db[l] + (1 - beta1) * self.db[l]
                self.W[l] -= learning_rate * self.V_dW[l]
                assert(self.b[l].shape == self.V_db[l].shape)
                self.b[l] -= learning_rate * self.V_db[l]
        elif optimizer == 'rms_prop':
            epsilon = 1e-8
            beta2 = kwargs.get('beta2', 0)
            for l in range(self.L)[1:]:
                '''print(
                    self.S_dW[l].shape,
                    self.dW[l].shape,
                    self.S_db[l].shape,
                    self.db[l].shape)
                    '''
                self.S_dW[l] = beta2 * self.S_dW[l] + \
                    (1 - beta2) * (self.dW[l] ** 2)
                self.S_db[l] = beta2 * self.S_db[l] + \
                    (1 - beta2) * (self.db[l] ** 2)
                self.W[l] -= learning_rate * \
                    (self.dW[l] / (np.sqrt(self.S_dW[l]) + epsilon))
                self.b[l] -= learning_rate * \
                    (self.db[l] / (np.sqrt(self.S_db[l]) + epsilon))
        elif optimizer == 'adam':
            epsilon = 1e-8
            beta1 = kwargs.get('beta1', 0)
            beta2 = kwargs.get('beta2', 0)

            for l in range(self.L)[1:]:
                '''print(
                    self.S_dW[l].shape,
                    self.dW[l].shape,
                    self.S_db[l].shape,
                    self.db[l].shape)'''
                self.V_dW[l] = beta1 * self.V_dW[l] + (1 - beta1) * self.dW[l]
                self.V_db[l] = beta1 * self.V_db[l] + (1 - beta1) * self.db[l]
                self.S_dW[l] = beta2 * self.S_dW[l] + \
                    (1 - beta2) * (self.dW[l] ** 2)
                self.S_db[l] = beta2 * self.S_db[l] + \
                    (1 - beta2) * (self.db[l] ** 2)
                self.V_dW_corr[l] = self.V_dW[l] / (1 - beta1 ** num_epoch)
                self.V_db_corr[l] = self.V_db[l] / (1 - beta1 ** num_epoch)
                self.S_dW_corr[l] = self.S_dW[l] / (1 - beta2 ** num_epoch)
                self.S_db_corr[l] = self.S_db[l] / (1 - beta2 ** num_epoch)
                self.W[l] -= learning_rate * \
                    (self.V_dW_corr[l] /
                     (np.sqrt(self.S_dW_corr[l]) + epsilon))
                self.b[l] -= learning_rate * \
                    (self.V_db_corr[l] /
                     (np.sqrt(self.S_db_corr[l]) + epsilon))

    def nn_model(
            self,
            learning_rate=0.01,
            num_epochs=100,
            mini_batch_size=10,
            lmbda=0,
            optimizer='gradient_descent'):
        ''' This method performs the gradient descent algorithm to
        minimize the cost function and find out the optimal values of the
        parameters W and b.

        Arguments:
        learning_rate : controls the rate which with the values of the
                        parameters change.
        num_epochs : number of epochs of training.
        mini_batch_size : mini-batch size for mini-batch gradient descent.
        optimizer : the optimization algorithm used to find parameters.

                    options :-

                    'gradient_descent' : usual gradient descent optimizer
                    'momentum' : gradient descent with momentum
                    'rms_prop' : gradient descent with RMSprop optimizer
                    'adam' : adam optimizer
        '''

        self.mini_batch_size = mini_batch_size
        self.lmbda = lmbda

        for epoch in range(num_epochs):
            self.mini_batches = self.create_mini_batches()

            for mini_batch in self.mini_batches:
                X, Y = mini_batch
                forward_cache = self.forward_propagation(
                    X, Y, output_activation='softmax')
                self.compute_cost(
                    forward_cache, lmbda, output_activation='softmax')
                self.backward_propagation(lmbda, forward_cache)
                self.update_parameters(
                    learning_rate, optimizer, epoch, beta1=0.9, beta2=0.999)

            print('Epoch', epoch, ':', self.cost)

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

    def store_parameters(self):
        ''' This method stores the parameters W and b in file for future use.'''

        L = self.L - 1
        filename = input('Enter filename : ')

        # store the file in output directory with .npz extension
        filename = '../output/' + filename + '.npz'

        stored_arrays = {'sizes': self.sizes}
        for l in range(L):
            stored_arrays['W' + str(l + 1)] = self.W[l + 1]
            stored_arrays['b' + str(l + 1)] = self.b[l + 1]

        np.savez(filename, **stored_arrays)

    def load_parameters(self, filename):
        ''' This method loads the parameters stored in the file 'filename'.

        Arguments:
        filename : complete path of the file in which parameters are stored
        '''

        L = self.L - 1
        parameters = np.load(filename)

        sizes = parameters['sizes']
        print('NN layer sizes:', sizes)

        for l in range(L):
            self.W[l + 1] = parameters['W' + str(l + 1)]
            self.b[l + 1] = parameters['b' + str(l + 1)]
