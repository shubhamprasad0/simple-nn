import numpy as np


def activation(Z, func_name):
    ''' This method applies the activation function specified by 'func_name'
    to 'Z' and returns the value obtained.

    Arguments:
    Z : real value or numpy array
    func_name : string denoting the activation function name.
                'sigmoid' : refers to the sigmoid activation function.
                'relu' : refers to the ReLU activation function.

    Returns:
    A : real value or numpy array of the same dimension as Z, obtained after
        applying the particular activation function to Z.
    '''

    if func_name == 'sigmoid':
        A = sigmoid(Z)
    elif func_name == 'relu':
        A = relu(Z)

    return A


def sigmoid(Z):
    ''' This method applies the sigmoid activation function to the input Z and
    returns the output as A.

    Arguments:
    Z : real value or numpy array

    Returns:
    A : real value or numpy array of the same dimension as Z, obtained after
        applying sigmoid activation to Z.
    '''

    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    ''' This method applies the ReLU activation function to the input Z and
    returns the output as A.

    Arguments:
    Z : real value or numpy array

    Returns:
    A : real value or numpy array of the same dimension as Z, obtained after
        applying ReLU activation to Z.
    '''

    A = np.maximum(0, Z)
    return A


def one_hot_encoding(Y, num_classes):
    ''' This method converts the given vector Y into one hot encoded form.

    Arguments:
    Y : numpy array of shape (1, m)
    num_classes : number of classes in the classification problem

    Returns:
    Y_one_hot : numpy array of shape(num_classes, m)
    '''

    Y_one_hot = np.eye(num_classes)[:, Y[0]]
    return Y_one_hot


def sigmoid_gradient(Z):
    ''' This method finds the derivative of the sigmoid activation function
    and finds and returns the value of the derivative for the input Z.

    Arguments:
    Z : real value or numpy array.

    Returns:
    sigma_gradient : value of the derivative of sigmoid function at Z.
    '''

    sigma = sigmoid(Z)
    sigma_gradient = sigma * (1 - sigma)
    return sigma_gradient


def relu_gradient(Z):
    ''' This method finds the derivative of the ReLU activation function and
    finds and returns the value of the derivative for the input Z.

    Arguments:
    Z : real value or numpy array.

    Returns:
    relu_gradient : value of the derivative of ReLU function at Z.
    '''

    relu_gradient = (Z > 0)
    return relu_gradient


def activation_gradient(Z, func_name):
    ''' This method applies the derivative of the activation function
    specified by 'func_name' to 'Z' and returns the value obtained.

    Arguments:
    Z : real value or numpy array
    func_name : string denoting the activation function name.
                'sigmoid' : refers to the sigmoid activation function.
                'relu' : refers to the ReLU activation function.

    Returns:
    A : real value or numpy array of the same dimension as Z, obtained after
        applying the particular derivative function to Z.
    '''

    if func_name == 'sigmoid':
        return sigmoid_gradient(Z)
    else:
        return relu_gradient(Z)


def one_hot_decoding(Y_hat):
    ''' This method converts the one-hot-encoded version of output labels, to
    normal (1, m) 1-D matrix form.

    Arguments:
    Y_hat : one-hot-encoded output labels of shape (NUM_CLASSES, m)

    Returns:
    Y : a matrix of output labels of shape (1, m)
    '''

    Y = np.argmax(Y_hat)
    return Y


def generate_output_file(Y_hat):
    ''' This file generates the output file in required format.

    Arguments:
    Y_hat : This matrix contains the predicted labels.
    '''

    with open('../output/predictions.csv', 'w') as outfile:
        outfile.write('ImageId,Label\n')
        for i, val in enumerate(Y_hat):
            outfile.write(str(i + 1) + ',' + val + '\n')
