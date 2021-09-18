import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import resize

digit = datasets.load_digits()
n_samples = len(digit.images)

target = digit.target
digits = digit.images

shapes=[8, 16, 64]
data=[]
for i in shapes:
    temp=[]
    for img in digits:
        temp.append(np.ravel(resize(img, (i, i), anti_aliasing=False)))
    data.append(np.array(temp))
    temp.clear()

split=[0.1, 0.2, 0.3, 0.4]
gamma = [0.1, 0.01, 0.001, 0.0001]
#gamma = [0.0001]
for g in gamma :
  print(f"\n=== Gamma = {g} ===\n")
  print("ImgSize\tTrn:Tst\tValAcc\tTstAcc")
  for d in data :
    for size in split :
      clf = svm.SVC(gamma=g)
      X_train, X_test, y_train, y_test = train_test_split(d, target, test_size=size, shuffle=False)
      X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=int(len(y_test)/2), shuffle=False)
      clf.fit(X_train, y_train)
      y_val_pred = clf.predict(X_val)
      y_pred = clf.predict(X_test)
      shape0 = int(math.sqrt(d[0].shape[0]))
      shape1 = int(math.sqrt(d[0].shape[0]))
      train = int((1-size)*100)
      test = int(size*100)
      valacc = accuracy_score(y_val, y_val_pred)*100
      testacc = accuracy_score(y_test, y_pred)*100
      print(f"{shape0}*{shape1}\t{train}:{test}\t{valacc:.2f}\t{testacc:.2f}")
    print()
  print()
