import sklearn
import pickle, os
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def resize_data(data, rescale_factor):
    resized_images=[]
    for img in data:
        resized_images.append(np.ravel(resize(img, (i, i), anti_aliasing=False)))
    return resized_images

def create_split(data, target, train_size, val_size):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=(1-train_size), shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size, shuffle=True)
    return (X_train, y_train, X_val, y_val, X_test, y_test)

def get_acc(model,X,Y):
    pred = model.predict(X)
    acc = metrics.accuracy_score(Y,pred)
    f1 = metrics.f1_score(Y, pred, average=None)
    return {"accuracy" : acc, "f1 score" : f1}

def runClassificationExample1(X_train, y_train, X_val, y_val, gamma, model_path, split):
    clf = svm.SVC(gamma = gamma)
    clf.fit(X_train, y_train)
    metrics_valid = get_acc(clf, X_val, y_val)
    if metrics_valid["accuracy"] < 0.6:
        print(f"Skipping model for gamma={gamma} valacc={metrics_valid['accuracy']}")
        return None
    name=f"./model-{split*10}-{gamma}-{metrics_valid['accuracy']:.4f}.sav"
    pickle.dump(clf, open(os.path.join(model_path, name), 'wb'))
    metrics_valid["model"] = clf
    return metrics_valid

def runClassificationExample2(X_train, y_train, X_val, y_val, model_path, clf_name, hyperparameter):
    clf = None
    if clf_name == 'svm':
        clf = svm.SVC(gamma=hyperparameter)
    elif clf_name == 'desctree':
        clf = DecisionTreeClassifier(max_depth=hyperparameter)
    else :
        return None
    clf.fit(X_train, y_train)
    metrics_valid = get_acc(clf, X_val, y_val)
    if metrics_valid['accuracy'] < 0.6:
        return None
    metrics_valid["model"] = clf
    return metrics_valid

def findBestModel(model_list):
    if len(model_list) == 0:
        return None
    best_model_accuracy = model_list[0]['accuracy']
    best_model = model_list[0]
    for i in model_list :
        if i['accuracy'] > best_model_accuracy :
            best_model_accuracy = i['accuracy']
            best_model = i
    return best_model