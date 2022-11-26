import numpy as np

from node import Node

class CartTree:
    def __init__(self):
        self.root = None
    
    def fit(self, x_data, y_data):
        self.root = Node(x_data, y_data)

    def _predict(self, x_test):
        return self.root.predict(x_test)

    def predict(self, x_test):
        if len(x_test.shape) == 1: # (3,)
            return self._predict(x_test)
        return np.array([self._predict(x) for x in x_test])
    
    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return sum(a == b for a, b in zip(y_pred, y_test)) / y_test.size
