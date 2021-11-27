from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
import numpy as np
from collections import defaultdict, OrderedDict
import pickle, os

def create_split(X, y):
    total_size = len(X)
    random_permutation = np.random.permutation(total_size)
    train_size = int(total_size*0.7)
    test_size = int((total_size - train_size)/2)
    val_size = test_size

    X_train = X[random_permutation[:train_size]]
    y_train = y[random_permutation[:train_size]]

    X_test = X[random_permutation[train_size:train_size+test_size]]
    y_test = y[random_permutation[train_size:train_size+test_size]]

    X_val = X[random_permutation[-test_size:]]
    y_val = y[random_permutation[-test_size:]]

    return X_train, y_train, X_val, y_val, X_test, y_test

iris = load_iris()
X = iris.data
y = iris.target

gamma = [0.001, 0.01, 0.1, 1, 10, 100]
print("\nGamma\tSplit Run-1\tSplit Run-2\tSplit Run-3\tTr-Mean\tTstMean\tValMean")
print("===============================================================================")
path = './models/'
if os.path.exists(path)==False:
    os.mkdir(path)

gamma_f1 = defaultdict(float)

for g in gamma :
    print(g, end='\t')
    clf = SVC(gamma=g)
    y_train_pred = []
    y_test_pred = []
    y_val_pred = []
    for _ in range(3):
        X_train, y_train, X_val, y_val, X_test, y_test = create_split(X, y)
        model = clf.fit(X_train, y_train)
        y_train_pred.append(f1_score(y_train, model.predict(X_train), average='macro'))
        y_test_pred.append(f1_score(y_test, model.predict(X_test), average='macro'))
        y_val_pred.append(f1_score(y_val, model.predict(X_val), average='macro'))
        print(f"{len(X_train)}:{len(X_val)}:{len(X_test)}", end='\t')

        name = f"{g}-{f1_score(y_test, model.predict(X_test), average='macro'):.3f}.sav"
        pickle.dump(model, os.path.join(path, name))

    print(f"{np.mean(y_train_pred):.2f}\t{np.mean(y_val_pred):.2f}\t{np.mean(y_test_pred):.2f}")
    gamma_f1[g] = (np.mean(y_test_pred) + np.mean(y_val_pred))/2

best_gamma = 0
best_f1 = 0
gamma_f1 = sorted(gamma_f1.items(), key=lambda d : (d[1], d[0]), reverse=True)

print(f"\nThe best hyperparameter (Gamma) is : {gamma_f1[0][0]}\nThe bad hyperparameters (Gamma) are : {gamma_f1[-1][0]} and {gamma_f1[-2][0]}\n")