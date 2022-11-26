from time import perf_counter
import cProfile, pstats

from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from tree import CartTree

if __name__ == "__main__":
    args_mk_cls = {
        "n_samples": 20000,
        "n_features": 10,
        "n_informative": 8,
        "n_redundant": 2,
        "n_classes": 4,
        "n_clusters_per_class": 1,
        "class_sep": 1.2,
    }

    x_array, y_array = make_classification(**args_mk_cls)
    #x_array, y_array = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x_array.astype(np.float32), y_array, test_size=0.33
    )

    #Custom Knn
    beg = perf_counter()
    
    profiler = cProfile.Profile()
    profiler.enable()

    tree = CartTree()
    tree.fit(x_train, y_train)
    
    trees = [[tree, 0.]]
    with tqdm() as pbar:
        while trees[-1][0].nb_leaf > 1:
            trees.append(trees[-1][0].prune_tree())
            pbar.update(1)
            
    #print("| nb_leaf | alpha | Acc Train | Acc test |")
    #for tree, alpha in trees:
    #    print(f"| {tree.nb_leaf:7d} | {alpha:.3f} |     {tree.score(x_train, y_train):.3f} |    {tree.score(x_test, y_test):.3f} |")
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profil")

    print(f"Custom CART & Pruning -> {perf_counter() - beg:.3f}s.")