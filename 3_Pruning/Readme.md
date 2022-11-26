# Partie 3 Pruning

## I. Préparation du terrain

Avant de rentrer dans le dur du code, reprenons rapidement les variables necessaires au prining. on de souvient que pour le pruning on devra calculer pour chaque noeud $g(t) =\frac{R(t) - R(T_t)}{|T_t| - 1}$ et retirer le noeud ayant le plus petit g.

### 1. Calcul de $|T_t|$ 

$|T_t|$ est le nombre de feuille partant de $t$, une écriture récursive semble évidente et assez simple. Cela donne :

```python
class Node:
    def nb_leaf(self):
        if self.left is None:
            return 1
        return self.left.nb_leaf() + self.right.nb_leaf()
```

Pour l'instant tout va bien.

### 2. Calcul de $R(t)$

$R(t)$ est l'erreur de classification si ce noeud était considéré une feuille. On peut récupérer cette information à l'initialisation du noeud. On va en profiter aussi pour stocker le label de classification, cela servira après pruning ce noeud devient vraiment une feuille.

On modifie donc notre __init__ :

```python
class Node:
    def __init__(self, x_data, y_data):
        count_labels = count(y_data, y_data.max() + 1)
        self.label = np.argmax(count_labels)
        self.miscla_error = 1 - count_labels.max()/y_data.size
        
        self.left = None
        self.right = None
        
        if self.miscla_error == 0.:
            return
        
        #Suite avec le split
```

Pour l'instant ca va aussi.

### 3. Calcul de $R(T_t)$

$R(T_t)$ est l'erreur de classification des données d'entrainement passées pas ce noeud. C'est soit son erreur de classification si c'est une feuille (celle calculé dans 2.), soit la moyenne pondérée par la proportion de donnée allant à gauche et à droite de $R(T_{t_{left}})$ et $(T_{t_{right}})$.

Afin de pouvoir pondérer notre moyenne on doit garder en mémoire combien de noeux partent de chaque coté, on rajoute cela à `__init__` (je montre pas le code cela devrait aller).

Reste plus qu'à écrire la méthode :

```python
class Node
    @property
    def miscla_tree(self):
        if self.left is None:
            return self.miscla_node
        return (
            self.size_data_left * self.left.miscla_tree 
            + self.size_data_right * self.right.miscla_tree
        ) / (self.size_data_left + self.size_data_right)
```

J'ai rajouté le `@property` devant qui sert globalement à me laisser appeler cette valeur avec `node.miscla_tree`plutôt que `node.miscla_tree()` qui me semble moins naturel. Nul besoin de vous expliquer les properties dans python mais je vous laisser lien de la doc si jamais vous êtes curieux : [property](https://docs.python.org/3/library/functions.html#property). J'en ai profité pour mettre `@property` devant `nb_leaf`aussi.

### 4. Itération

La on va toucher à un point un peu sensible, forcément on va avoir envie d'itérer sur nos noeuds, hors ils sont rangés dans un arbre ce qui ne simplifie pas les choses. Il y a de nombreuses manières de le faire mais il y en a une qui pour moi est supérieure, elle est en revanche assez avancée et n'est pas forcément abordable. Je vais vous montrer une méthode "moche" et ma méthode que nous garderons, elle sont casiment équivalente et j'expliquerai en quoi la mienne diffère pour que vous puissiez jouer avec le code si vous le souhaitez.

### 4.1. Méthode "moche"

On va tout simplement créer une liste avec tout nos noeuds et les ajouter un par un récursivement.

```python
class CartTree:
    def get_all_nodes(self):
        res = []
        self.root.get_all_nodes(res)
        return res

class Node:
    def get_all_nodes(self, nodes = None):
        nodes.append(self)
        if self.left is not None:
            self.left.get_all_nodes(nodes)
            self.right.get_all_nodes(nodes)
```

On crée dans `CartTree.get_all_nodes` la liste qui va servir à stocker tout nos noeuds, ensuite on les ajoute un par un dans notre liste que l'on passe par référence à la méthode. Cela fonctionne car une liste passé en paramètre peut être modifiée dans la fonction, ce sont les mêmes objets.

Je suis pas fan car cela nous oblige à créer une liste entière avec tout nos noeuds alors qu'itérer est la seule chose que l'on veut, y accéder un par un.

### 4.2. Méthode générateur

Les générateur en python sont semblables aux fonctions, sauf que ce coup ci on ne `return` pas une valeur mais on en `yield` une ou plusieurs. A chaque fois que notre code rencontrera un `yield` il se met en pause et attend la prochaine itération pour reprendre son cours et s'arreter au prochain yield. 

Pour vous faire un exemple simple : 

```python 
def gen():
    for k in range(10):
        yield k**2

for x in gen():
    print(x, end=' ')

# output : 0 1 4 9 16 25 ...
```

Le `yield from` indique juste qu'il faudra `yield` tout les éléments de l'itérable indiqué juste après. 

Je sais, l'explication est obscure, il me faudrait un autre repo pour vous expliquer en profondeur les générateur et leur puissance. Dans les faits je vais juste obtenir un objet sur lequel je peux itérer et qui me donnera les noeuds un par un.

Voici le résultat : 

```python
class Node:
    def gen_all_nodes(self):
        yield self
        if self.left is not None:
            yield from self.left.gen_all_nodes()
            yield from self.right.gen_all_nodes()
```

Et du coup on a tout ce dont on a besoin pour notre pruning sans trop forcer, plus qu'à l'implémenter.

## II. Pruning

Pour le pruning, on va se contenter d'écrire une méthode `prune_tree` dans la class `CartTree` qui nous renverra le prochain arbre pruné, on aura plus qu'à l'appeler en boucle jusqu'à pruning complet pour avoir notre pruning complet.

On va commencer par faire une deepcopy de notre arbre, on en crée un nouveau completement identique qu'on pourra élaguer sans perdre l'ancien. On initialise aussi les variables necessaires à la recherche du $\alpha$ minimum.

```python
from copy import deepcopy

class CartTree:
    def prune_tree(self):
        tree = deepcopy(self)
        min_alpha = float("inf")
        best_node = None
```

On itère ensuite sur nos noeuds pour calculer $g(t$)$ et garder le meilleur.

```python
class CartTree:
    def prune_tree(self):
        ...
        for node in tree.root.gen_all_nodes():
            if node.nb_leaf == 1:
                continue
            g = (node.miscla_node - node.miscla_tree) / (node.nb_leaf - 1)
            if g < min_g:
                min_g = g
                best_node = node
        
```

Nous reste plus qu'à retirer le noeud et return le nouvel arbre avec sa valeur de $\alpha$. 

```python
class CartTree:
def prune_tree(self):
        ...       
        tree.root.remove_node(best_node)
        return tree, min_g
```

On va implémenter rapidement la méthode `Node.remove_node`.

```python
class CartTree:
    def remove_node(self, node):
        if self is node:
            self.left = None
            self.right = None
        elif self.left is not None:
            self.left.remove_node(node)
            self.right.remove_node(node)
```

On devrait être bon avec tout ca ! Regardons ca avec `test.py`:

**Output test.py**

| nb_leaf | alpha | Acc Train | Acc test |
| ------: | ----: | --------: | -------: |
|      17 | 0.000 |     1.000 |    0.920 |
|      11 | 0.002 |     0.992 |    0.931 |
|      10 | 0.005 |     0.990 |    0.936 |
|       4 | 0.013 |     0.940 |    0.904 |
|       2 | 0.023 |     0.924 |    0.910 |
|       1 | 0.304 |     0.619 |    0.644 |