from time import perf_counter
import cProfile, pstats

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

from tree import CartTree

if __name__ == "__main__":
    args_mk_cls = {
        "n_samples": 4000,
        "n_features": 4,
        "n_informative": 4,
        "n_redundant": 0,
        "n_classes": 3,
        "n_clusters_per_class": 2,
        "class_sep": 1.2,
    }

    x_array, y_array = make_classification(**args_mk_cls)

    x_train, x_test, y_train, y_test = train_test_split(
        x_array.astype(np.float32), y_array, test_size=0.33
    )

    #Custom Knn
    beg = perf_counter()
    
    profiler = cProfile.Profile()
    profiler.enable()

    tree = CartTree()
    tree.fit(x_train, y_train)
    acc = tree.score(x_test, y_test)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profil")
    
    print()

    print(f"Custom CART -> {perf_counter() - beg:.3f}s.")
    print(f"   Accuracy -> {acc:.3f}.")