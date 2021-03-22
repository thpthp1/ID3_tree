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

class Node:
    def __init__(self, label, is_leaf=False, min_sample_leaf=1):
        self.label = label
        self.is_leaf = is_leaf
        self.children = []
        self.min_sample_leaf = min_sample_leaf
        self.attr_idx = None
        self.val = None
        self.eval_func = None
        self.type = None # 'cont' or 'cat'

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __str__(self) -> str:
        return str(self.__dict__)

    def _train(self, X, y: np.ndarray, properties, prop_idx):
        """
        X: attrs
        y: classes
        properties: list of propaerties of each attr
            ['cat', 'cont', 'cat', 'cont']
        """
        #print(properties)
        assert X.shape[0] == y.size
        classes, counts = np.unique(y, return_counts=True)
        # Apply root to be labelled as the most popular class
        most_pop = mode(y)[0][0]
        if len(y) <= self.min_sample_leaf or classes.size == 1 or len(properties) <= 0: return Node(most_pop, is_leaf=True)
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
                # One single unique value, cannot split
                if col_sort.size <= 1:
                    continue
                split_points = (col_sort[1::] + col_sort[0:-1]) * 0.5
                possible_i_res = np.zeros_like(split_points)
                index = 0
                for sp in split_points:
                    y_hi, y_lo = y[col > sp], y[col <= sp]
                    for D in [y_hi, y_lo]:
                        __, cond_counts = np.unique(D, return_counts=True)
                        cond_class_entropy = entropy(cond_counts/D.size, base=2)
                        possible_i_res[index] += D.size/ y.size * cond_class_entropy
                    index += 1
                try:
                    I_res_idx= np.argmin(possible_i_res)
                except ValueError:
                    print(y)
                    print(col_sort)
                    print(possible_i_res)
                    raise Exception
                I_res = possible_i_res[I_res_idx]
                possible_known_vals = np.array([split_points[I_res_idx]])
            if I - I_res >= IG:
                best_attr = i
                known_vals = possible_known_vals
                IG = I - I_res
        self.type = properties[best_attr]
        self.attr_idx = prop_idx[best_attr]
        if self.type == 'cat':
            root_val = []
            for val in known_vals:
                root_val.append(val)
                x_sub, y_sub = X[X[:, best_attr] == val], y[X[:, best_attr] == val]
                if y_sub.size <= 0:
                    self.children.append(Node(most_pop, is_leaf=True))
                else:
                    self.children.append(Node(None)._train(
                                        np.delete(x_sub, best_attr, axis=1),
                                        y_sub,
                                        properties=properties[:best_attr] + properties[best_attr+1:],
                                        prop_idx=prop_idx[:best_attr] + prop_idx[best_attr+1:]))
            self.val = root_val
        elif self.type == 'cont':
            val = known_vals[0] # single Y/N split
            self.val = val
            x_val_hi, y_val_hi = X[X[:, best_attr] > val], y[X[:, best_attr] > val]
            x_val_lo, y_val_lo = X[X[:, best_attr] <= val], y[X[:, best_attr] <= val]
            if y_val_lo.size <= 0:
                self.children.append(Node(most_pop, is_leaf=True))
            else:
                self.children.append(Node(None)._train(
                                        np.delete(x_val_lo, 1, axis=1),
                                        y_val_lo,
                                        properties=properties[:best_attr] + properties[best_attr+1:],
                                        prop_idx=prop_idx[:best_attr] + prop_idx[best_attr+1:]))
            if y_val_hi.size <= 0:
                self.children.append(Node(most_pop, is_leaf=True))
            else:
                self.children.append(Node(None)._train(
                                        np.delete(x_val_hi, best_attr, axis=1),
                                        y_val_hi,
                                        properties=properties[:best_attr] + properties[best_attr+1:],
                                        prop_idx=prop_idx[:best_attr] + prop_idx[best_attr+1:]))
        return self

    @classmethod
    def train(cls, X, y, properties):
        return Node(None)._train(X, y, properties, list(range(len(properties))))





class DecisionTreeClassifier:

    def __init__(self, min_sample_leaf=1) -> None:
        self.min_sample_leaf = min_sample_leaf
        self.root = Node(None)


    def train(self, X, y, properties=[]):
        if not properties: properties = self._attr_types(X)
        self.root = Node.train(X, y, properties)

    def _attr_types(self, X, cat_thresh=3):
        properties = []
        for i in range(X.shape[1]):
            if np.unique(X[:, i]).size > cat_thresh:
                properties.append('cont')
            else: properties.append('cat')
        return properties

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
            label = DecisionTreeClassifier._traverse(vec, self.root)
            pred = np.append(pred, label)
        return pred

    @classmethod
    def _traverse(cls, vec, r: Node):
        if r.is_leaf:
            return r.label
        else:
            best_val = vec[r.attr_idx]
            if r.type == 'cont':
                if best_val <= r.val:
                    return DecisionTreeClassifier._traverse(vec, r.children[0])
                else:
                    return DecisionTreeClassifier._traverse(vec, r.children[1])
            else:
                #print(r.val)
                idx = r.val.index(best_val)
                return DecisionTreeClassifier._traverse(vec, r=r.children[idx])

def formatTree(t,s,labels=["outlook", "temp", "humidity", "windy"]):
    if not t.is_leaf:
        print("----"*s+ "attr: {}, vals: {}, idx: {}".format(labels[t.attr_idx], t.val, t.attr_idx))
    else:
        print("----"*s + "label: {}, vals: {}, idx: {}".format(t.label, t.val, t.attr_idx))
    for child in t.children:
        formatTree(child,s+1,labels=labels)

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
    node = Node.train(X, Y, ["cat", "cont", 'cont', 'cat'])
    #print(node.children[0])
    #formatTree(node, 0)
    
    trans_data = pd.get_dummies(df[['outlook', 'temp', 'humidity', 'windy']], drop_first=True)
    baseline = DTC(criterion="entropy")
    baseline.fit(trans_data.iloc[:-1], Y)
    print("Baseline prediction: ", baseline.predict(trans_data.iloc[-1].to_numpy().reshape(1, -1)))
    model = DecisionTreeClassifier()
    model.train(trans_data.iloc[:-1].to_numpy(), Y, properties=["cont", "cont", 'cont', 'cont', 'cont'])
    formatTree(model.root, 0, labels=trans_data.columns.tolist())
    print("Impl predict: ", model.predict(trans_data.iloc[-1]))
    plot_tree(baseline, feature_names=trans_data.columns)
    plt.show()
