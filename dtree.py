import numpy as np
from pandas.core.indexes import base
import scipy
from scipy.stats import mode, entropy
import pandas as pd
from pprint import pprint
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

DEBUG = False

def int_info(a: np.ndarray):
    return np.sum(-np.log2(a/a.size)*a/a.size)

def gini_index(a: np.ndarray):
    return 1 - np.sum(a**2)

class Node:
    def __init__(self, label, is_leaf=False, min_sample_leaf=1):
        self.label = label
        self.is_leaf = is_leaf
        self.children = []
        self.min_sample_leaf = min_sample_leaf
        self.attr_idx = None
        self.val = None
        self.type = None # 'cont' or 'cat'
        self.info = None

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __str__(self) -> str:
        return str(self.__dict__)

    def _train(self, X, y: np.ndarray, properties, prop_idx, method='entropy', algo='ID3'):
        """
        X: attrs
        y: classes
        properties: list of propaerties of each attr
            ['cat', 'cont', 'cat', 'cont']
        method: {'entropy', 'gini', 'gain_ratio'}
        algo: {'ID3', 'CART'}
        """
        gain_func = self._gain_func(method)
        #print(properties)
        assert X.shape[0] == y.size
        classes, counts = np.unique(y, return_counts=True)
        # Apply root to be labelled as the most popular class
        most_pop = mode(y)[0][0]
        if len(y) <= self.min_sample_leaf or classes.size == 1 or len(properties) <= 0: 
            if DEBUG:
                print(f"Hit leaf node on single sample: {len(y) <= self.min_sample_leaf},\
                same class: {classes.size == 1} with class {classes[0]}, no more attr: {len(properties) <= 0}")
            return Node(most_pop, is_leaf=True)
        I = gain_func(counts/y.size)
        self.info = I

        if DEBUG: print("Current Node info: ", I)
        """
        For each column, calculate the entropy of the column of data
        """
        best_attr = None
        known_vals = None
        best_attr, known_vals = self._best_attr(X, y, I, prop_idx, properties, method)
        self.type = properties[best_attr]
        self.attr_idx = prop_idx[best_attr]
        if self.type == 'cat':
            root_val = []
            for val in known_vals:
                root_val.append(val)
                x_sub, y_sub = X[X[:, best_attr] == val], y[X[:, best_attr] == val]
                self._add_branch(x_sub, y_sub, best_attr, properties, prop_idx, method, algo)
            self.val = root_val
        elif self.type == 'cont':
            val = known_vals[0] # single Y/N split
            self.val = val
            x_val_hi, y_val_hi = X[X[:, best_attr] > val], y[X[:, best_attr] > val]
            x_val_lo, y_val_lo = X[X[:, best_attr] <= val], y[X[:, best_attr] <= val]
            self._add_branch(x_val_lo, y_val_lo, best_attr, properties, prop_idx, method, algo)
            self._add_branch(x_val_hi, y_val_hi, best_attr, properties, prop_idx, method, algo)
        return self

    def _best_attr(self, X, y, I, prop_idx, properties, method):
        gain = 0
        for i in range(X.shape[1]):
            # calculate gain
            col = X[:, i]
            possible_known_vals = None
            if properties[i] == 'cat':
                new_gain, possible_known_vals = self._information_cat(col, y, I, method)
            elif properties[i] == 'cont':
                col_sort = np.sort(np.unique(col))
                # One single unique value, cannot split
                if col_sort.size <= 1:
                    continue
                new_gain, possible_known_vals = self._information_cont(col, y, I, method)
            if DEBUG: 
                    print(
                        f"New gain found with gain: {new_gain} I_res: {I - new_gain} for idx: {prop_idx[i]} with known_vals: {possible_known_vals}"
                        )
            if new_gain >= gain:
                best_attr = i
                known_vals = possible_known_vals
                gain = new_gain
        return best_attr, known_vals

    def _add_branch(self, x, y, best_attr, properties, prop_idx, method, algo):
        most_pop = mode(y)[0][0]
        if algo == 'ID3':
            new_prop = properties[:best_attr] + properties[best_attr+1:]
            new_prop_idx = prop_idx[:best_attr] + prop_idx[best_attr+1:]
            new_x = np.delete(x, best_attr, axis=1)
        else:
            new_prop = properties
            new_prop_idx = prop_idx
            new_x = x
        if y.size <= 0:
            self.children.append(Node(most_pop, is_leaf=True))
        else:
            self.children.append(Node(None)._train(
                                    new_x,
                                    y,
                                    properties=new_prop,
                                    prop_idx=new_prop_idx,
                                    method=method,
                                    algo=algo))

    def _information_cont(self, col, y, info, method='entropy'):
        gain_func = self._gain_func(method)
        col_sort = np.sort(np.unique(col))
        split_points = (col_sort[1::] + col_sort[0:-1]) * 0.5
        possible_i_res = np.zeros_like(split_points)
        index = 0
        for sp in split_points:
            y_hi, y_lo = y[col > sp], y[col <= sp]
            for D in [y_hi, y_lo]:
                __, cond_counts = np.unique(D, return_counts=True)
                cond_class_entropy = gain_func(cond_counts/D.size)
                possible_i_res[index] += D.size/ y.size * cond_class_entropy
            index += 1
        i_res_idx= np.argmin(possible_i_res)
        I_res = possible_i_res[i_res_idx]
        best_split = split_points[i_res_idx]
        possible_known_vals = np.array([best_split])
        val_counts = np.array([np.sum(col > best_split), np.sum(col <= best_split)])
        if method == "gain_ratio":
            gain = (info - I_res) / int_info(val_counts/y.size)
        else: gain = info - I_res
        return gain, possible_known_vals

    def _information_cat(self, col, y, info, method='entropy'):
        gain_func = self._gain_func(method)
        I_res = 0
        possible_known_vals, val_counts = np.unique(col, return_counts=True)
        for val, val_count in zip(possible_known_vals, val_counts):
            y_val = y[col == val]
            __, cond_counts = np.unique(y_val, return_counts=True)
            cond_class_entropy = gain_func(cond_counts/y_val.size)
            I_res += val_count/ y.size * cond_class_entropy
        if method == "gain_ratio":
            gain = (info - I_res) / int_info(val_counts/y.size)
        else: gain = info - I_res
        return gain, possible_known_vals

    def _gain_func(self, method):
        if method == 'gini':
            gain_func = lambda p: gini_index(p)
        else:
            gain_func = lambda p: entropy(p, base=2)
        return gain_func

    def format_tree(self, s, labels):
        if not self.is_leaf:
            print("----"*s+ "attr: {}, vals: {}, info: {}".format(labels[self.attr_idx], self.val, self.info))
        else:
            print("----"*s + "Leaf label: {}".format(self.label))
        for child in self.children:
            child.format_tree(s+1,labels)

    @classmethod
    def train(cls, X, y, properties, method='entropy', algo="ID3"):
        return Node(None)._train(X, y, properties, list(range(len(properties))), method=method, algo=algo)

class DecisionTreeClassifier:

    def __init__(self, min_sample_leaf=1) -> None:
        self.min_sample_leaf = min_sample_leaf
        self.root = Node(None)


    def train(self, X, y, properties=[], method='entropy', algo='ID3'):
        if not properties: properties = self._attr_types(X)
        self.root = Node.train(X, y, properties, method=method, algo=algo)

    def _attr_types(self, X, cat_thresh=3):
        properties = []
        for i in range(X.shape[1]):
            if np.unique(X[:, i]).size > cat_thresh:
                properties.append('cont')
            else: properties.append('cat')
        return properties

    def format_tree(self, labels):
        self.root.format_tree(0, labels)

    def predict(self, X):
        if X.ndim < 2:
            X_mat = np.array([X])
        else: X_mat = np.array(X)
        assert X_mat.ndim == 2
        pred = np.array([])
        # print(X_mat)
        for i in range(X_mat.shape[0]):
            vec = X_mat[i]
            # print(vec)
            label = self._traverse(vec)
            pred = np.append(pred, label)
        return pred

    def _traverse(self, vec):
        curr = self.root
        while not curr.is_leaf:
            best_val = vec[curr.attr_idx]
            if curr.type == 'cont':
                if best_val <= curr.val:
                    curr = curr.children[0]
                else:
                    curr = curr.children[1]
            else:
                #print(r.val)
                idx = curr.val.index(best_val)
                curr = curr.children[idx]
        return curr.label

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    X = df.drop("play",axis=1).to_numpy()
    Y = df.play.to_numpy()
    test_val = pd.DataFrame(
        {
            "outlook": ["overcast"],
            "temp": [60],
            "humidity": [62],
            "windy": [False]
        }
    )
    df = df.append(test_val, ignore_index=True)
    # node = Node.train(X, Y, ["cat", "cont", 'cont', 'cat'])
    #print(node.children[0])
    # format_tree(node, 0, ["outlook", "temp", "humidity", "windy"])
    trans_data = pd.get_dummies(df[['outlook', 'temp', 'humidity', 'windy']], drop_first=True)
    baseline = DTC(criterion="entropy")
    baseline.fit(trans_data.iloc[:-1], Y)
    print("Baseline prediction: ", baseline.predict(trans_data.iloc[-1].to_numpy().reshape(1, -1)))
    model = DecisionTreeClassifier()
    model.train(trans_data.iloc[:-1].to_numpy(), Y, properties=["cont", "cont", 'cont', 'cont', 'cont'], method='entropy', algo='CART')
    model.format_tree(trans_data.columns.tolist())
    print("Impl predict: ", model.predict(trans_data.iloc[-1]))
    plot_tree(baseline, feature_names=trans_data.columns)
    plt.show()
