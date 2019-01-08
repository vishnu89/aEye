'''
This a simple example of feed forward and back propagation network using python
'''
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def getPlot(nrows, ncols, returnFig=False):
    fig, axplts = plt.subplots(nrows, ncols)
    if nrows > 1 or ncols > 1:
        axplts = axplts.ravel()
    else:
        axplts = [axplts]
    if returnFig:
        return fig, axplts
    else:
        return axplts


class SigmoidLayer:
    def __init__(self):
        pass

    def forward(self, inData, weight):
        outData = 1.0 / (1.0 + np.exp(-inData))
        return outData

    def backward(self, delta, weight):
        # derivative calculation
        outDelta = delta
        return outDelta * (1.0 - outDelta)


class TanhLayer:
    def __init__(self):
        pass

    def forward(self, inData, weight):
        return np.tanh(inData)

    def backward(self, delta, weight):
        # derivative calculation
        return 1.0 - delta ** 2


class Perceptron:
    def __init__(self):
        pass

    def forward(self, inData, weight):
        return np.dot(inData, weight)

    def backward(self, delta, weight):
        return np.dot(delta, weight.T)


class SumOfSquaredCostFunction:
    def __init__(self):
        pass

    def forward(self, data, actual):
        sqErr = np.sum(((data - actual) ** 2) ** .5)
        # sqErr = abs(data - actual)
        return sqErr

    def backward(self, output, actual):
        # derivative
        return -1 * (output - actual)


class Layer:
    def __init__(self, weight, learningRate, linearLayer, activationLayer):
        self.weight = weight
        self.inData = None
        self.outData = None
        self.linear = linearLayer()
        self.nonLinear = activationLayer()
        self.learningRate = learningRate

    def forward(self, data):
        self.inData = data
        data = self.linear.forward(self.inData, self.weight)
        data = self.nonLinear.forward(data, self.weight)
        self.outData = data
        return self.outData

    def backward(self, indelta):
        indelta = indelta * self.nonLinear.backward(self.outData, self.weight)
        outdelta = self.linear.backward(indelta, self.weight)
        # update weight
        update_data = np.atleast_2d(self.inData)
        update_delta = np.atleast_2d(indelta)
        self.weight = self.weight + self.learningRate * np.dot(update_data.T, update_delta)
        return outdelta


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = []
        activation = SigmoidLayer
        networkSize = []
        for inLayer, outLayer in zip(layers, layers[1:]):
            networkSize.append([inLayer + 1, outLayer + 1])  # for bias
        networkSize[-1][-1] -= 1  # last layer should not have bias
        for layerSize in networkSize:
            r = 2 * np.random.random(layerSize) - 1
            layer = Layer(r, .2, Perceptron, activation)
            self.layers.append(layer)
        # for layer in self.layers:
        #     print layer.weight
        # print '______________________________________________'

    def fit(self, X, y, epochs=200000, printSize=10000):
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        axplts = getPlot(1, 1)[0]
        costFn = SumOfSquaredCostFunction()
        errors = []
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            data, actual = X[i], y[i]
            for layer in self.layers:
                data = layer.forward(data)
            # output layer
            delta = costFn.backward(self.layers[-1].outData, actual)
            for layer in self.layers[::-1]:  # traverse in reverse for back propagation
                delta = layer.backward(delta)
            if k % printSize == 0:
                sqErr = costFn.forward(self.layers[-1].outData, actual)
                print 'epochs: k, sqErr', k, sqErr
                errors.append((k, sqErr))
        x, y = zip(*errors)
        axplts.plot(x, y)

    def predict(self, datas):
        resDatas = []
        for odata in datas:
            data = np.concatenate((np.ones(1).T, np.array(odata)), axis=-1)
            for layer in self.layers:
                data = layer.forward(data)
            resDatas.append(data.tolist())
            print odata, data
        return resDatas


def main():
    # model = NeuralNetwork([2, 3, 5, 2, 3, 1])
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([1, 0, 0, 1])
    # model.fit(X, y)
    # model.predict(X)
    #
    # model = NeuralNetwork([3, 3, 5, 2, 3, 1])
    # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]])
    # y = np.array([1, 0, 1, 0])
    # model.fit(X, y)
    # for e in X:
    #     print(e, model.predict(e))

    nn = NeuralNetwork([2, 2, 2])
    X = np.array([[0.05, 0.10]])
    y = np.array([[0.01, .99]])
    nn.fit(X, y)
    nn.predict(X)

    # nn = NeuralNetwork([2, 5, 1])
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 1, 1, 0])
    # nn.fit(X, y)
    # print nn.predict(X)


if __name__ == '__main__':
    main()
    plt.show()
