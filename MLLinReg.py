from sklearn.neural_network import MLPRegressor
import numpy as np
def trainModel(dataset):
    data = np.array(dataset)
    X = data[:, :-1]
    y = data[:, -1]

    model = MLPRegressor(
        hidden_layer_sizes=(34, 34),
        activation='relu',
        max_iter=2000
    )
    model.fit(X, y)
    return model