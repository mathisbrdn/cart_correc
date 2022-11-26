import numpy as np
from utils import *

class Node:
    def __init__(self, x_data, y_data):
        if  np.unique(y_data).size == 1:
            self.left = None
            self.right = None
            self.label = y_data[0]
            return

        self.dim_split, self.val_split = find_best_split(x_data, y_data, y_data.max() + 1)
        
        if self.dim_split == np.inf:
            self.left = None
            self.right = None
            self.label = np.argmax(count(y_data, y_data.max() + 1))
            return
        
        self.left = Node(
            x_data[x_data[:, self.dim_split] < self.val_split],
            y_data[x_data[:, self.dim_split] < self.val_split],
        )
        
        self.right = Node(
            x_data[x_data[:, self.dim_split] >= self.val_split],
            y_data[x_data[:, self.dim_split] >= self.val_split],
        )
        
    def predict(self, x_test):
        if self.left is None:
            return self.label
        if x_test[self.dim_split] < self.val_split:
            return self.left.predict(x_test)
        return self.right.predict(x_test)
 