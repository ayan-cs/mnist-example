import math
from utils import *
from sklearn.datasets import load_iris

def test_create_split():
    n = 10
    iris = load_iris()
    X = iris.data[:n]
    y = iris.target[:n]
    X_train, y_train, X_val, y_val, X_test, y_test = create_split(X, y, 0.7, 0.1)
    assert len(X_train) == int(0.7 * len(X))
    assert len(X_test) == math.ceil(0.2 * len(X))
    assert len(X_val) == math.ceil(0.1 * len(X))
    assert len(X_train)+len(X_test)+len(X_val) == n
