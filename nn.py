import numpy as np
import matplotlib.pyplot as plt

from myFuncTools import *



class Layer:
    def __init__(self, units=8, activation='relu', inputShape=None) -> None:
        self.units = units
        self.activation = activation
        self.inputShape = inputShape
        self.params = {}

    def activate(self, layerInput):
        cache = {}
        cache["Z"] = np.dot(self.params["W"], layerInput) + self.params["b"]
        match self.activation:
            case "relu":
                cache["A"] = np.maximum(cache["Z"], 0)
            case "sigmoid":
                cache["A"] = 1 / (1 + np.exp(-cache["Z"]))
            case "softmax":
                buffArr = cache["Z"] - np.amax(cache["Z"], axis=0, keepdims=True)
                exp_buffArr = np.exp(buffArr)
                sum_exp_buffArr = np.sum(exp_buffArr, axis=0, keepdims=True)

                cache["A"] = exp_buffArr / sum_exp_buffArr  
            case _:
                cache["A"] = cache["Z"]
        
        return cache["Z"], cache["A"]
    
    def backwardProp(self, gradA, Aprev):
        # A = activate(Z)
        # Z = W*Aprev + b
        Z, A = self.activate(Aprev)
        m = Aprev.shape[1]
        grad = {}
        match self.activation:
            case "relu":
                grad["dZ"] = gradA * np.array(Z > 0, dtype="int32")
            case "sigmoid":
                grad["dZ"] = gradA * (A * (1-A))
            case "softmax":
                grad["dZ"] = A * (gradA - np.sum(gradA * A, axis=0, keepdims=True))
            case _:
                grad["dZ"] = np.ones(Z.shape)

        grad["dW"] = (1/m) * np.dot(grad["dZ"], Aprev.T)
        grad["db"] = (1/m) * np.sum(grad["dZ"], axis=1, keepdims=True)
        dAprev = np.dot(self.params["W"].T, gradA)
        return grad, dAprev

    def __str__(self) -> str:
        return f"Layer:\nUnits: {self.units}, activation: {self.activation}, input shape: {self.inputShape}\n"
    




class Model:
    def __init__(self, numberLayer=0, layers=None) -> None:
        self.numberLayer = numberLayer
        self.layers = layers
        self.optimization = None
        self.loss = None
        self.metrics = None
        self.learningRate = 10e-3

        # For hyperparameter tunning
        self.bestParams = None

        # Store the result
        self.costs = None

    def add(self, newLayer):
        if self.layers == None: self.layers = []
        self.layers.append(newLayer)
        self.numberLayer += 1
    
    def _initLayersParams(self, X_train):
        self.layers[0].params["W"] = np.random.rand(self.layers[0].units, X_train.shape[0])
        self.layers[0].params["b"] = np.zeros((self.layers[0].units, 1))
        for i in range(1, self.numberLayer):
            self.layers[i].params["W"] = np.random.rand(self.layers[i].units, self.layers[i-1].units)
            self.layers[i].params["b"] = np.zeros((self.layers[i].units, 1))

    def _forwardPropagation(self, X_input):
        caches = {}
        caches["A0"] = X_input
        for i in range(0, self.numberLayer):
            caches["Z"+str(i+1)], caches["A"+str(i+1)] = self.layers[i].activate(caches["A"+str(i)])

        return caches
    
    def _cost(self, yhat, y):
        # y is the actual value
        # yhat is the value predicted

        m = y.shape[1]

        match self.loss:
            case "categorical_crossentropy":
                return (-1/m) * np.sum((y * np.log(yhat)) + ((1-y) * np.log(1-yhat)))
            case "quadratic":
                return (1/m) * np.sum(np.power(y - yhat, 2))
            
    def _backwardPorpagation(self, caches, y):
        grads = {}
        m = y.shape[1]

        match self.loss:
            case "categorical_crossentropy":
                grads["dA"+str(self.numberLayer)] = (-1/m) * ((y / (caches["A"+str(self.numberLayer)])) - ((1-y) / (1-caches["A"+str(self.numberLayer)])))
            case "quadratic":
                grads["dA"+str(self.numberLayer)] = (2/m) * (caches["A"+str(self.numberLayer)] - y)
        
        for i in range(self.numberLayer, 0, -1):
            buffGrad, dAprev = self.layers[i-1].backwardProp(grads["dA"+str(i)], caches["A"+str(i-1)])

            grads["dW"+str(i)] = buffGrad["dW"]
            grads["db"+str(i)] = buffGrad["db"]
            if i != 1: grads["dA"+str(i-1)] = dAprev
        
        return grads

    def _updateParams(self, grads, learning_rate):
        match self.optimization:
            case "gds":
                for j in range(self.numberLayer):
                    self.layers[j].params["W"] -= learning_rate * grads["dW"+str(j+1)]
                    self.layers[j].params["b"] -= learning_rate * grads["db"+str(j+1)]

    def _metric(self, y_pred, y):
        match self.metrics:
            case "accuracy":
                return accuracy(y_pred, y)
        
            
    def compile(self, optimization, loss, metrics):
        self.optimization = optimization
        self.loss = loss
        self.metrics = metrics

    def fit(self, X_train, y_train, validation_data=None, batch_size=0, epochs=20, autoHyperTunning=False):
        
        X_train = X_train.T
        y_train = y_train.T

        batchs = []
        if (batch_size > 0):
            batchNumber = (X_train.shape[0] // batch_size) +1
            for i in range(0, batchNumber-1):
                batchs.append((X_train[:, batch_size*i, batch_size*[i+1]], y_train[:, batch_size*i, batch_size*[i+1]]))
            batchs.append((X_train[:, batch_size*(batchNumber-1):], y_train[:, batch_size*(batchNumber-1):]))
        else:
            batchs.append((X_train, y_train))

        

        if validation_data != None:
            X_valid = validation_data[0].T
            y_valid = validation_data[1].T

        # learning rate tunning
        if autoHyperTunning:
            learing_rate_range = np.arange(-5, -1, 0.1)
            accuracies = []
            bestParams = []
            for i in learing_rate_range:
                learning_rate = 10 ** i
                # init params
                self._initLayersParams(X_train)

                for i in range(epochs):
                    for batch in batchs:
                        current_X_train = batch[0]
                        current_y_train = batch[1]

                        # forward propagation
                        caches = self._forwardPropagation(current_X_train)
                        
                        # backward propagation
                        grads = self._backwardPorpagation(caches, current_y_train)

                        # update params
                        self._updateParams(grads, learning_rate)
                        
                currentAccuracy = 0.3*self._metric(self.predict(X_valid.T), y_valid.T) + 0.7*self._metric(self.predict(X_train.T), y_train.T)
                if (len(accuracies) == 0 or currentAccuracy >= np.max(accuracies)): bestParams.append(learning_rate)
                accuracies.append(currentAccuracy)

            self.bestParams = bestParams
            self.learningRate = bestParams[-1]


        # final train

        # init params
        self._initLayersParams(X_train)

        costs = []

        for i in range(epochs):
            for batch in batchs:
                current_X_train = batch[0]
                current_y_train = batch[1]

                # forward propagation
                caches = self._forwardPropagation(current_X_train)

                # current cost
                cost = self._cost(caches["A"+str(self.numberLayer)], y_train)
                costs.append(cost)
                
                # backward propagation
                grads = self._backwardPorpagation(caches, current_y_train)

                # update params
                self._updateParams(grads, learning_rate)

            print (f"+++ Epoch {i} +++")
            print(f"Train accuracy: {self._metric(self.predict(X_train.T), y_train.T):.2f}, train loss: {cost:.2f}")
            if validation_data != None:
                lossCaches = self._forwardPropagation(X_valid)
                loss = self._cost(lossCaches["A"+str(self.numberLayer)], y_valid)
                print(f"Validation accuracy: {self._metric(self.predict(X_valid.T), y_valid.T):.2f}, validation loss: {loss:.2f}")
        
        self.costs = costs
        

    def predict(self, X_test):
        caches = self._forwardPropagation(X_test.T)
        y_pred_prob = caches["A"+str(self.numberLayer)].T
        y_pred = np.zeros_like(y_pred_prob)
        for i in range(len(y_pred_prob)):
            max_index = np.argmax(y_pred_prob[i])
            y_pred[i][max_index] = 1
        return y_pred
    
    def showTrainingCost(self):
        if self.costs == None:
            print("You have not train the model")
        else:
            plt.plot(range(len(self.costs)), self.costs)
            plt.title("Training cost")
            plt.xlabel("Epoch")
            plt.ylabel("Cost")
            plt.show()
        
    def __str__(self) -> str:
        myStr = f"model:\nnumber of layer: {self.numberLayer}\n"
        for i in self.layers:
            myStr += str(i)
        return myStr


