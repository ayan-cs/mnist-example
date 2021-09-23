import sklearn
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split
def resize_data(data, rescale_factor):
    resized_images=[]
    for img in data:
        resized_images.append(np.ravel(resize(img, (i, i), anti_aliasing=False)))
    return resized_images

def create_split(data, target, train_size, val_size):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=(1-train_size), shuffle=False)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size, shuffle=False)
    return (X_train, y_train, X_val, y_val, X_test, y_test)

def get_acc(model,X,Y):
    pred = model.predict(X)
    acc = metrics.accuracy_score(Y,pred)
    return acc
