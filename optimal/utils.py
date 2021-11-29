import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix



def create_train_test_split(X,Y,train_size = 0.8):
    test_size = 1-train_size
    assert train_size + test_size == 1
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=train_size)
    return X_train,X_test,Y_train,Y_test

def g_SVM_metrics(train_X,train_Y,test_X,test_Y,hyperparameter):
    clf = SVC(gamma=hyperparameter)
    clf.fit(train_X,train_Y)
    predict = clf.predict(test_X)
    acc = accuracy_score(y_true=test_Y,y_pred=predict)
    f1 = f1_score(y_true=test_Y,y_pred=predict,average='macro')
    return {'acc':acc,'f1':f1},clf

def g_test_metrics(model,test_x,test_y):
    predict = model.predict(test_x)
    acc = accuracy_score(y_true=test_y,y_pred=predict)
    f1 = f1_score(y_true=test_y,y_pred=predict,average='macro')
    return {'acc':acc,'f1':f1},predict