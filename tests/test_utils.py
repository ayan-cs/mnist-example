import math
from sklearn.datasets import load_iris
import os
from mnist_savemodel import *

# Take the runclassification() and check if it is being saved
def test_save_model():
    model_path = './models'
    iris = load_iris()
    random_indice = np.random.permutation(len(iris.data))
    X = iris.data[random_indice]
    y = iris.target[random_indice]
    acc = run_classification_experiment(X[:50, :], y[:50], X[50:100, :], y[50:100], .01, .2)
    assert os.path.exists(model_path)

def test_small_data_overfit():
    iris = load_iris()
    random_indice = np.random.permutation(len(iris.data))
    X = iris.data[random_indice[:10]]
    y = iris.target[random_indice[:10]]
    acc = run_classification_experiment(X, y, X, y, .001, 1)
    assert acc > 0.99