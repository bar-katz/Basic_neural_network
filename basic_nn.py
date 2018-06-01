# Bar Katz 208561357

import numpy as np
import pickle
import time


def read():
    """
    Read data from a serialized file
    :return: train data and data to test on
    """
    data = open(r"C:\Users\Bar\PycharmProjects\ml_ex3\me.txt", "rb").read()
    c = pickle.loads(data)
    train_x = c[0]
    train_y = c[1]
    return train_x, train_y


def leakyrelu(x):
    """
    leakyReLU function
    :param x: vector to activate function on
    :return: the vector after activation of function
    """
    for i in range(len(x)):
        x[i] = max(0.01 * x[i], x[i])
    return x


def leakyrelu_tag(x):
    """
    leakyReLU function derivative
    :param x: vector to activate function derivative on
    :return: the vector after activation of function derivative
    """
    for i in range(len(x)):
        x[i] = max(0.01 * np.sign(x[i]), np.sign(x[i]))
    return x


def softmax(w):
    """
    softmax function
    :param w: vector to activate softmax function on
    :return: the vector after activation of softmax function
    """
    e = np.exp(np.array(w) - np.max(w))
    return e / np.sum(e)


def tanh(x):
    """
    tanh function
    :param x: vector to activate function on
    :return: the vector after activation of function
    """
    return np.tanh(x)


def tanh_tag(x):
    """
    tanh function derivative
    :param x: vector to activate function derivative on
    :return: the vector after activation of function derivative
    """
    ones = np.ones(x.shape)
    return ones - np.tanh(x) ** 2


def forward(x, model, params={}):
    """
    Forward propagation of x input on the network
    :param x: input sample
    :param model: network parameters
    :param params: a dictionary to use in case you need forward propagation parameters
    :return: vector of probabilities - to what class does the input belong
    """
    w1, b1, w2, b2 = [model[key] for key in ('W1', 'b1', 'W2', 'b2')]
    active_func = active['func']

    z1 = np.dot(w1, x) + b1
    h1 = active_func(z1)
    z2 = np.dot(w2, h1) + b2
    y_hat = softmax(z2)

    params['z1'], params['h1'], params['z2'], params['h2'] = z1, h1, z2, y_hat
    return y_hat


def backward(x, y, params):
    """
    Backward propagation of x input on the network
    :param x: input sample
    :param y: input class
    :param params: needed parameters to preform backward propagation
    :return: derivatives of the network parameters on the loss function(multiclass cross-entropy)
    """
    y_hat, h1, w2, z1 = [params[key] for key in ('y_hat', 'h1', 'w2', 'z1')]
    active_deri = active['derivative']

    dz2 = y_hat
    dz2[y] -= 1
    dw2 = np.dot(dz2, h1.T)
    db2 = dz2

    dz1 = np.dot(y_hat.T, w2).T * active_deri(z1)
    dw1 = np.dot(dz1, x.T)
    db1 = dz1

    return dw1, db1, dw2, db2


def update(model, deri, lr):
    """
    Update network parameters by the computed back propagation derivatives
    :param model: network parameters
    :param deri: derivatives computed in back propagation phase
    :param lr: learning rate
    :return: updated model
    """
    w1, b1, w2, b2 = [model[key] for key in ('W1', 'b1', 'W2', 'b2')]
    dw1, db1, dw2, db2 = [deri[key] for key in ('dW1', 'db1', 'dW2', 'db2')]

    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

    model = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}
    return model


# activation functions
activation_funcs = {'leakyReLU': {'func': leakyrelu, 'derivative': leakyrelu_tag},
                    'tanh': {'func': tanh, 'derivative': tanh_tag}}

lr = 0.001
hidden_neurons_size = 100
epochs = 10
active = activation_funcs['tanh']


def main():
    """
    Main
    """

    w1 = np.random.uniform(low=-0.08, high=0.08, size=(hidden_neurons_size, 784))
    b1 = np.random.uniform(low=-0.24, high=0.24, size=(hidden_neurons_size, 1))

    w2 = np.random.uniform(low=-0.23, high=0.23, size=(10, hidden_neurons_size))
    b2 = np.random.uniform(low=-0.73, high=0.73, size=(10, 1))

    model = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}

    # read data
    train_x, train_y = read()

    sample_size = train_x.shape[0]

    # shuffle samples
    s = np.arange(train_x.shape[0])
    np.random.shuffle(s)
    train_x = train_x[s]
    train_y = train_y[s]

    # divide data - 80% train, 20% validation
    train_x, vali_x = train_x[:int(sample_size * 0.8), :], train_x[int(sample_size * 0.8):, :]
    train_y, vali_y = train_y[:int(sample_size * 0.8)], train_y[int(sample_size * 0.8):]

    for epoch in range(epochs):
        # shuffle samples
        s = np.arange(train_x.shape[0])
        np.random.shuffle(s)
        train_x = train_x[s]
        train_y = train_y[s]

        loss = 0
        vali_loss = 0
        vali_scc = 0

        start = time.clock()
        # train
        for x, y in zip(train_x, train_y):
            # normalize the vector
            x = np.ndarray.astype(x, dtype=float)
            x /= 255
            x = np.expand_dims(x, axis=1)

            # forward
            back_params = {}
            y_hat = forward(x, model, back_params)
            h1, z1 = back_params['h1'], back_params['z1']

            # compute loss
            loss -= np.log(y_hat[y])

            # backward
            params = {'y_hat': y_hat, 'h1': h1, 'w2': model['W2'], 'z1': z1}
            dw1, db1, dw2, db2 = backward(x, y, params)

            # update
            deri = {'dW1': dw1, 'db1': db1, 'dW2': dw2, 'db2': db2}
            model = update(model, deri, lr)

        # validation
        for x, y in zip(vali_x, vali_y):

            # normalize the vector
            x = np.ndarray.astype(x, dtype=float)
            x /= 255
            x = np.expand_dims(x, axis=1)

            # forward
            y_hat = forward(x, model)

            # compute success
            if y == np.argmax(y_hat):
                vali_scc += 1

            # compute loss
            vali_loss -= np.log(y_hat[y])

        end = time.clock()
        avg_loss = loss / (sample_size * 0.8)
        avg_vali_loss = vali_loss / (sample_size * 0.2)
        vali_acc = vali_scc / (sample_size * 0.2)

        print '#', epoch, ' acc ', vali_acc, ' loss ', avg_loss[0], ' vali loss ', avg_vali_loss[0], ' time ', \
            (end - start)


if __name__ == "__main__":
    main()
