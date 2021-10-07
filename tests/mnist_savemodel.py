# Best model finding, modularizing functionalities

import matplotlib.pyplot as plt
import numpy as np
import math
import os

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import resize
import pickle
from utils import resize_data, create_split, get_acc, run_classification_experiment

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

for g in gamma :
    for size in split :
        X_train, y_train, X_val, y_val, X_test, y_test = create_split(data, target, size, (1-size)/2)
        acc = run_classification_experiment(X_train, y_train, X_val, y_val, g, size)
        #testacc = get_acc(model = clf, X = X_test, Y = y_test )*100
    print()

import glob
all_models=glob.glob('./models/*.sav')
max_valacc = all_models[0].split('-')[-1].split('.')[1]
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