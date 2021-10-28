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

split=[0.1, 0.2, 0.3, 0.4]
gamma = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
best_valacc=0
model=[]

model_path="./models"
if os.path.exists(model_path)==False:
    os.mkdir(model_path)

candidate_model=[]
#best_model={}
best_valacc = 0
for g in gamma :
    #print(f"\n=== Gamma = {g} ===\n")#
    #print("ImgSize\tTrn:Tst\tValAcc")#
    for size in split :
        #clf = svm.SVC(gamma=g)
        X_train, y_train, X_val, y_val, X_test, y_test = create_split(data, target, size, (1-size)/2)
        metrics_valid = runClassificationExample1(X_train, y_train, X_val, y_val, g, model_path, size)
        """if metrics_valid["accuracy"]>best_valacc:
        best_valacc=valacc
        best_model['gamma']=g
        best_model['clf']=clf
        best_model['testsplit']=size
        best_model['Validation Acc']=valacc"""
        if metrics_valid :
            candidate = {
                "model" : metrics_valid["model"],
                "accuracy" : metrics_valid["accuracy"],
                "f1 score" : metrics_valid["f1 score"],
                "gamma" : g,
                "test split" : split
            }
            candidate_model.append(candidate)
            #print(candidate)

import glob
all_models=glob.glob('./models/*.sav')
max_valacc = all_models[0].split('-')[-1].split('.')[0]
best_model = all_models[0]
for name in all_models:
    valacc = name.split('-')[-1].split('.')[1]
    if valacc>max_valacc:
        max_valacc=valacc
        best_model=name

print("Best model from hard drive :\n",best_model)
model = pickle.load(open(best_model, 'rb'))
y_pred = model.predict(X_test)
print(f"Accuracy score from loaded model {best_model} : ", accuracy_score(y_test, y_pred)*100)

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