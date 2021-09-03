import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
n_samples = len(digits.images)
print("Image size",digits.images[0].shape)
data = digits.images.reshape((n_samples, -1))

print("Image Size\tTrain:Test\tAccuracy")
split=[0.1, 0.2, 0.3]
for size in split :
  clf = svm.SVC(gamma=0.001)
  X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=size, shuffle=False)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(digits.images[0].shape,'\t',int((1-size)*100),':',int(size*100),'\t',accuracy_score(y_test, y_pred))
