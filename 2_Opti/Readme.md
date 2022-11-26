# Partie 2 : Optimisation

On l'a vu dans la partie 0, le code est lent. Par "lent" j'entends : trop lent à mon goût ET on voit qu'il y a moyen d'améliorer le tout sans se casser les deux bras. On sait grâce au profiling que la partie trop lente se situe dans [utils.py](https://github.com/mathisbrdn/correc_cart/1_Construction/utils.py), et principalement sur le `Coutner`, essayons de recoder cela.

> Attention : J'utilise numba pour l'optimisation, je ne vous recommande pas de l'utiliser comme reflèxe. Je l'utilise dans ce cas car je sais que sur ce genre d'opération les résultats seront là. De plus, numba peut être très contraignant à utiliser on essaye donc de limiter son utilisation uniquement aux cas critiques. La biblihotèque peut faire des merveilles sur les sur les manipulations d'arrays mais évitez d'aller plus loin. Je vous invite à lire la doc si vous voulez vous y intéresser : [Numba](https://numba.readthedocs.io/en/stable/index.html).

Tout simplement on va remplacer notre `Counter` par une nouvelle fonction qui fait exactement la même chose : 

```python
@njit
def count(labels, nb_labels):
    count = np.zeros((nb_labels,), dtype = np.float32)
    for lab in labels:
        count[lab] += 1.
    return count

count(np.zeros(1,dtype=np.int),2)
```

On en profite pour réecrire gini :

```python
@njit
def gini(labels, nb_labels):
    res = 1.
    for c in count(labels, nb_labels):
        res -= (c/labels.size)**2
    return res

gini(np.zeros(1,dtype=np.int),2)
```

Tout simple. On peut déja voir les problèmes liés à numba, on va devoir lui passer en paramètres le nombre de classes afin qui n'ait pas à itérer lui même sur le array en amont, ce qui va charge la syntax et on est obligé de préciser chaque type utilisé et on devra se limiter à ces types à chaque appel mais on verra ca plus bas encore.

La dernière ligne est un appel simple de la fonction pour qu'il puisse la compiler à l'import de 'utils.py' et pas pendant notre construction d'arbre.

La question c'est maintenant de savoir si cela change quelque chose dans nos performances ! Je vous passes les petits changements dans liés à la nouvelle syntax de count.

**Résultat -> 1.1s**

C'est bien mais pas fou encore, on sait que cela peut valoir le coup de modifier un peu `find_best_split` pour éviter d'itérer encore et encore sur nos labels. Nos données sont triées à chaque fois selon l'axe en cours de tests. Entre chaque test de split il n'y a donc qu'un label qui change de côté, essayons de prendre cela en compte :

```python
@njit
def find_best_split(x_data, y_data, nb_labels):
    min_gini = np.inf
    dim_split = np.iinfo(np.int64).max #inf
    val_split = np.inf
    
    for dim in range(x_data.shape[1]):
        arg_sort = np.argsort(x_data[:,dim])
        x_data = x_data[arg_sort]
        y_data = y_data[arg_sort]

        #Initialisation de nos compteurs gauche et droite
        countl = np.zeros(nb_labels, dtype = np.float32)
        countl[y_data[0]] = 1.
        countr = count(y_data[1:], nb_labels)

        for k in range(1, arg_sort.size):
            #Calcul de gini devient en O(1), merci les compteurs
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

            #Update des compteurs
            countl[y_data[k]] += 1.
            countr[y_data[k]] -= 1.

    return dim_split, val_split

find_best_split(
    np.array([[1.,2.,3.]], dtype=np.float32),
    np.array([1,0,1], dtype=np.int),
    2
)
```

**Résultat -> 0.068s**

On commence à être pas mal du tout, on prend un facteur casi 200 sur notre cas et même plus grand sur des gros jeux de données. C'est peut être possible de faire mieux, j'en vois pas l'utilité. On reste sur un exercice et si le but est de faire l'arbre le plus rapide possible il vaudrait mieux descendre sur les langages compilés où on pourrait gérer manuellement la mémoire.

On va juste rajouter un petit truc par sécurité qui change pas grand chose mais peut éviter 2/3 soucis, si `find_best_split`, ne trouve pas de split on considère que le noeud est une feuille (c'est surtout pour le cas où tout les points sont égaux mais pas forcement de même labels).

On a donc sur notre `Node.__init__`

```python
class Node:
    def __init__(self, x_data, y_data):
        #Du code

        self.dim_split, self.val_split = find_best_split(x_data, y_data, y_data.max() + 1)
        
        if self.dim_split == np.inf:
            self.left = None
            self.right = None
            self.label = np.argmax(count(y_data, y_data.max() + 1))
            return
        
        #Encore du code
```
        

Ca sera tout pour cette partie.