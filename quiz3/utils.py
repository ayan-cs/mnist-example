import pickle
import os
from sklearn.metrics import accuracy_score

best_svm = './best_svm.sav'
best_dtree = './best_dtree.sav'

def svm_load_predict(test_X, test_y):
    svm = pickle.load(open(best_svm, 'rb'))
    y_pred = svm.predict(test_X)
    acc = accuracy_score([test_y], y_pred)
    return y_pred, acc

def dtree_load_predict(test_X, test_y):
    dtree = pickle.load(open(best_dtree, 'rb'))
    y_pred = dtree.predict(test_X)
    acc = accuracy_score([test_y], y_pred)
    return y_pred, acc
