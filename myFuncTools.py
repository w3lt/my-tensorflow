import numpy as np

def train_validation_test(X, y, validation_size, test_size, random_state):
    np.random.seed(random_state)
    
    indices = np.random.permutation(len(y))

    train_split_idx = int((1-validation_size-test_size)*len(y))
    valid_split_idx = int(validation_size*len(y))

    train_idx = indices[:train_split_idx]
    validation_idx = indices[train_split_idx:valid_split_idx+train_split_idx]
    test_idx = indices[valid_split_idx+train_split_idx:]

    X_train, X_validation, X_test = X[train_idx], X[validation_idx], X[test_idx]
    y_train, y_validation, y_test = y[train_idx], y[validation_idx], y[test_idx]

    return (X_train, y_train), (X_validation, y_validation), (X_test, y_test)

def confusionMatrix(ypred, y):
    confMatrix = np.zeros((y.shape[1], y.shape[1]))
    for i in range(len(y)):
        confMatrix[np.where(y[i] == 1)[0][0], np.where(ypred[i] == 1)[0][0]] += 1

    return(confMatrix)

def accuracy(ypred, y):
    confMatrix = confusionMatrix(ypred, y)
    return np.trace(confMatrix) / np.sum(confMatrix)