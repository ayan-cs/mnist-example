# Best model finding, modularizing functionalities

import matplotlib.pyplot as plt
import numpy as np
import math
import os

from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import resize
import pickle
from utils import *

digit = datasets.load_digits()
n_samples = len(digit.images)

target = digit.target
digits = digit.images

data=[]
for img in digits:
    data.append(np.ravel(img))
data = np.array(data)

split=0.3
gamma = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
depth = range(1, 11)
best_valacc=0
model=[]

model_path="./models"
if os.path.exists(model_path)==False:
    os.mkdir(model_path)

candidate_svm = []
candidate_desctree = []
#best_model={}
best_valacc = 0
print("Splitidx\tSVM_G\tDT_D\n")
for s in range(5):
    X_train, y_train, X_val, y_val, X_test, y_test = create_split(data, target, split, (1-split)/2)
    for g in gamma :
        metrics_valid = runClassificationExample2(X_train, y_train, X_val, y_val, model_path, 'svm', g)
        if metrics_valid :
            candidate = {
                "model" : metrics_valid["model"],
                "accuracy" : metrics_valid["accuracy"],
                "f1 score" : metrics_valid["f1 score"],
                "gamma" : g,
                "test split" : split
            }
            candidate_svm.append(candidate)
    best_svm = findBestModel(candidate_svm)
    for d in depth :
        metrics_valid = runClassificationExample2(X_train, y_train, X_val, y_val, model_path, 'desctree', d)
        if metrics_valid :
            candidate = {
                "model" : metrics_valid["model"],
                "accuracy" : metrics_valid["accuracy"],
                "f1 score" : metrics_valid["f1 score"],
                "depth" : d,
                "test split" : split
            }
            candidate_desctree.append(candidate)
    best_desctree = findBestModel(candidate_desctree)
    print(f"Split__{s}\t{best_svm['gamma']}\t{best_desctree['depth']}")

"""
best_model_memory = candidate_model[0]['model']
best_model_accuracy = candidate_model[0]['accuracy']
g = candidate_model[0]['gamma']
for i in candidate_model :
    if i['accuracy'] > best_model_accuracy :
        best_model_accuracy = i['accuracy']
        best_model_memory = i['model']
        g = i['gamma']

y_pred = best_model_memory.predict(X_test)
print(f"Accuracy score from best model in candidate list {best_model_memory} with gamma={g} : ", accuracy_score(y_test, y_pred)*100)
"""