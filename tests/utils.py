import matplotlib.pyplot as plt
import numpy as np
import math
import os

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import resize
import pickle

def resize_data(data, rescale_factor):
    resized_images=[]
    for img in data:
        resized_images.append(np.ravel(resize(img, (i, i), anti_aliasing=False)))
    return resized_images

def create_split(data, target, train_size, val_size):
    test_size = 1 - (train_size + val_size)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=(1-train_size), shuffle=False)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size/(val_size + test_size), shuffle=False)
    return (X_train, y_train, X_val, y_val, X_test, y_test)

def get_acc(model,X,Y):
    pred = model.predict(X)
    acc = metrics.accuracy_score(Y,pred)
    return acc

def run_classification_experiment(X_train, y_train, X_val, y_val, gamma, size):
    model_path = './models'
    clf = svm.SVC(gamma=gamma)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    valacc = get_acc(model = clf, X = X_val, Y = y_val)
    #if valacc < 0.6:
    #    print(f"Skipping model for gamma={gamma} valacc={valacc}")
    #    return
    name=f'./model-{size}-{gamma}-{valacc:.4f}.sav'
    pickle.dump(clf, open(os.path.join(model_path, name), 'wb'))
    return valacc