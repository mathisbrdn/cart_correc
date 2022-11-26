import numpy as np
from utils import *

class Node:
    def __init__(self, x_data, y_data):
        count_labels = count(y_data, y_data.max() + 1)
        self.label = np.argmax(count_labels)
        self.miscla_node = 1 - count_labels.max()/y_data.size
        
        self.left = None
        self.right = None
        
        if self.miscla_node == 0.:
            return

        self.dim_split, self.val_split = find_best_split(x_data, y_data, y_data.max() + 1)
        
        if self.dim_split == np.inf:  
            return
        
        self.size_data_left = y_data[x_data[:, self.dim_split] < self.val_split].size
        self.size_data_right = y_data.size - self.size_data_left
        
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
    
    @property
    def nb_leaf(self):
        if self.left is None:
            return 1
        return self.left.nb_leaf + self.right.nb_leaf
    
    @property
    def miscla_tree(self):
        if self.left is None:
            return self.miscla_node
        return (
            self.size_data_left * self.left.miscla_tree 
            + self.size_data_right * self.right.miscla_tree
        ) / (self.size_data_left + self.size_data_right)
        
    def gen_all_nodes(self):
        yield self
        if self.left is not None:
            yield from self.left.gen_all_nodes()
            yield from self.right.gen_all_nodes()
            
    def remove_node(self, node):
        if self is node:
            self.left = None
            self.right = None
        elif self.left is not None:
            self.left.remove_node(node)
            self.right.remove_node(node)