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
from utils import resize_data, create_split, get_acc

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

best_model={}
for g in gamma :
    #print(f"\n=== Gamma = {g} ===\n")#
    #print("ImgSize\tTrn:Tst\tValAcc")#
    for size in split :
        clf = svm.SVC(gamma=g)
        X_train, y_train, X_val, y_val, X_test, y_test = create_split(data, target, size, (1-size)/2)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        y_pred = clf.predict(X_test)#
        shape0 = int(math.sqrt(data[0].shape[0]))#
        shape1 = int(math.sqrt(data[0].shape[0]))#
        train = int((1-size)*100)#
        test = int(size*100)#
        valacc = get_acc(model = clf, X = X_val, Y = y_val )
        if valacc < 0.6:
            print(f"Skipping model for gamma={g} valacc={valacc}")
            continue
        if valacc>best_valacc:
            best_valacc=valacc
            best_model['gamma']=g
            best_model['clf']=clf
            best_model['testsplit']=size
            best_model['Validation Acc']=valacc
        name=f'./model-{size*10}-{g}-{valacc:.4f}.sav'
        pickle.dump(clf, open(os.path.join(model_path, name), 'wb'))
        testacc = get_acc(model = clf, X = X_test, Y = y_test )*100#
        #print(f"{shape0}*{shape1}\t{train}:{test}\t{valacc:.2f}\t{testacc:.2f}")#
    print()

print("Best model from memory : \n",best_model)

"""
best_gamma = max(gamma[0])
for g in range(len(gamma)-1):
    if max(model[g])>best_gamma:
        best_gamma = g
print("Best gamma : ",best_gamma)
"""

"""
print("Best model : ",best_model)
y_pred = best_model['clf'].predict(X_test)
print("Best model's Test acc : ",accuracy_score(y_test, y_pred)*100)
"""

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
