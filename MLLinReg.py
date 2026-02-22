import numpy as np
from sklearn.linear_model import LinearRegression

def trainModel(dataset):
    data = np.array(dataset)

    X = data[:, 0:5]
    y = data[:, 5]

    model = LinearRegression()
    model.fit(X, y)

    return model
