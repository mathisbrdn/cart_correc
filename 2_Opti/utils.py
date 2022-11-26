from collections import Counter
import numpy as np
from numba import njit

@njit
def count(labels, nb_labels):
    count = np.zeros((nb_labels,), dtype = np.float32)
    for lab in labels:
        count[lab] += 1.
    return count

count(np.zeros(1,dtype=np.int),2)

@njit
def find_best_split(x_data, y_data, nb_labels):
    min_gini = np.inf
    dim_split = np.iinfo(np.int64).max #inf
    val_split = np.inf
    
    for dim in range(x_data.shape[1]):
        arg_sort = np.argsort(x_data[:,dim])
        x_data = x_data[arg_sort]
        y_data = y_data[arg_sort]

        countl = np.zeros(nb_labels, dtype = np.float32)
        countl[y_data[0]] = 1.
        countr = count(y_data[1:], nb_labels)

        for k in range(1, arg_sort.size):
            
            gini_split = (
                (1 - np.sum((countl/k)**2))
                * k
                + (1 - np.sum((countr/((y_data.size - k)))**2))
                * (y_data.size - k)
            )
            if gini_split < min_gini:
                min_gini = gini_split
                dim_split = dim
                val_split = x_data[k, dim]

            countl[y_data[k]] += 1.
            countr[y_data[k]] -= 1.

    return dim_split, val_split

find_best_split(
    np.array([[1.,2.,3.]], dtype=np.float32),
    np.array([1,0,1], dtype=np.int),
    2
)