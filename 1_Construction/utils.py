from collections import Counter
import numpy as np

def gini(y_data):
    count = Counter(y_data)
    return 1 - sum((x/y_data.size)**2 for x in count.values())

def find_best_split(x_data, y_data):
    min_gini = float("inf")
    dim_split = None
    val_split = None
    
    for dim in range(x_data.shape[1]):
        arg_sort = np.argsort(x_data[:, dim])
        x_data = x_data[arg_sort]
        y_data = y_data[arg_sort]
        
        for x in x_data[1:, dim]:
            label_left = y_data[x_data[:,dim] < x]
            gini_left = gini(label_left)
            label_right = y_data[x_data[:,dim] >= x]
            gini_right = gini(label_right)
            
            gini_now = label_left.size * gini_left + label_right.size * gini_right
            
            if gini_now < min_gini:
                min_gini = gini_now
                dim_split = dim
                val_split = x
                
    return dim_split, val_split