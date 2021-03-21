import numpy as np
import scipy
from scipy.stats import mode, entropy

class Node:
    def __init__(self, label, is_leaf=False, min_sample_leaf=1):
        self.label = label
        self.is_leaf = is_leaf
        self.children = []
        self.min_sample_leaf = min_sample_leaf
        self.attr_idx = None
        self.val = None
        self.eval_funcs = []
        self.type = None # 'cont' or 'cat'
    
    def train(self, X, y: np.ndarray, properties):
        """
        X: attrs
        y: classes
        properties: list of propaerties of each attr
            ['cat', 'cont', 'cat', 'cont']
        """
        classes, counts = np.unique(y, return_counts=True)
        # Apply root to be labelled as the most popular class
        most_pop = mode(y)[0][0]
        if len(y) <= self.min_sample_leaf: return Node(most_pop, is_leaf=True)
        I = entropy(counts/y.size, base=2)
        """
        For each column, calculate the entropy of the column of data
        """
        IG = 0
        best_attr = None
        known_vals = None
        for i in range(X.shape[1]):
            # calculate IG
            col = X[:, i]
            I_res = 0
            possible_known_vals = None
            if properties[i] == 'cat':
                possible_known_vals, val_counts = np.unique(col, return_counts=True)
                for val, val_count in zip(possible_known_vals, val_counts):
                    y_val = y[col == val]
                    __, cond_counts = np.unique(y_val, return_counts=True)
                    cond_class_entropy = entropy(cond_counts/y_val.size, base=2)
                    I_res += val_count/ y.size * cond_class_entropy
            elif properties[i] == 'cont':
                col_sort = np.sort(np.unique(col))
                split_points = (col_sort[1::] + col_sort[0::]) * 0.5
                possible_i_res = np.zeros_like(split_points)
                index = 0
                for sp in split_points:
                    y_hi, y_lo = y[col > sp], y[col <= sp]
                    for D in [y_hi, y_lo]:
                        __, cond_counts = np.unique(D, return_counts=True)
                        cond_class_entropy = entropy(cond_counts/D.size, base=2)
                        possible_i_res[index] += D.size/ y.size * cond_class_entropy
                    index += 1
                I_res_idx= np.argmin(possible_i_res)
                I_res = possible_i_res[I_res_idx]
                possible_known_vals = np.array([split_points[I_res_idx]])
            if I - I_res >= IG:
                best_attr = i
                known_vals = possible_known_vals
                IG = I - I_res
        root = Node(None)
        if root.type == 'cat':
            for val in known_vals:
                root.type = properties[best_attr]
                root.attr_idx = best_attr
                x_sub, y_sub = X[:, X[best_attr] == val], y[X[best_attr] == val]
                branch = Node(None)
                if y_sub.size <= 0:
                    
        elif root.type == 'cont':
            val = known_vals[0] # single Y/N split
            x_val_hi, y_val_hi = X[:, X[best_attr] > val], y[X[best_attr] > val]
            x_val_lo, y_val_lo = X[:, X[best_attr] <= val], y[X[best_attr] <= val]
            
        return root

            

        

class DecisionTreeClassifier:

    def __init__(self, min_sample_leaf=1) -> None:
        self.min_sample_leaf = min_sample_leaf
        self.root = None
    

    def predict(self, X):
        pass