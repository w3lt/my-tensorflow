import numpy as np
import matplotlib.pyplot as plt
import nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
  
# Init the model
model = nn.Model() 
model.add(nn.Layer(units=8, activation='relu'))
model.add(nn.Layer(units=3, activation='softmax'))

# Compile model
model.compile(optimization="gds", loss="quadratic", metrics="accuracy")

# import dataset
dataset = pd.read_csv("iris.data")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.reshape(-1, 1)

# Preprocessing the label (y) of dataset
transformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder="passthrough")
y = transformer.fit_transform(y)

# Split train/validation/test set
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = nn.train_validation_test(X, y, validation_size=0.15, test_size=0.15, random_state=42)

# train model
model.fit(X_train, y_train,
          validation_data=(X_valid, y_valid),
          epochs=3000, autoHyperTunning=True)

# show training cost
model.showTrainingCost()

# predict on whole dataset
y_pred = model.predict(X)
print(nn.confusionMatrix(y_pred, y))
print(nn.accuracy(y_pred, y))

# predict on test dataset
y_pred = model.predict(X_test)
print(nn.confusionMatrix(y_pred, y_test))
print(nn.accuracy(y_pred, y_test))