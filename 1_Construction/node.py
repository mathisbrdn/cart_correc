from utils import *

class Node:
    def __init__(self, x_data, y_data):
        if gini(y_data) == 0:
            self.left = None
            self.right = None
            self.label = y_data[0]
            return

        self.dim_split, self.val_split = find_best_split(x_data, y_data)
        
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