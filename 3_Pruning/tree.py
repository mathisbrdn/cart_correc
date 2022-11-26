from copy import deepcopy

from tqdm import tqdm
import numpy as np

from node import Node

class CartTree:
    def __init__(self):
        self.root = None
    
    def fit(self, x_data, y_data):
        self.root = Node(x_data, y_data)
        
    def prune_tree(self):
        tree = deepcopy(self)
        min_g = float("inf")
        best_node = None
        
        for node in tree.root.gen_all_nodes():
            if node.nb_leaf == 1:
                continue
            g = (node.miscla_node - node.miscla_tree) / (node.nb_leaf - 1)
            if g < min_g:
                min_g = g
                best_node = node

        tree.remove_node(best_node)
        return tree, min_g
    
    @property
    def nb_leaf(self):
        return self.root.nb_leaf
    
    def remove_node(self, node):
        self.root.remove_node(node)
        
    def _predict(self, x_test):
        return self.root.predict(x_test)

    def predict(self, x_test):
        if len(x_test.shape) == 1: # (3,)
            return self._predict(x_test)
        return np.array([self._predict(x) for x in x_test])
    
    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return sum(a == b for a, b in zip(y_pred, y_test)) / y_test.size