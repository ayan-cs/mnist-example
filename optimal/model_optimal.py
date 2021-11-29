import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import os
from utils import create_train_test_split,g_SVM_metrics,g_test_metrics

digits = load_digits()

n_sample = len(digits.images)
data = digits.images.reshape((n_sample,-1))

train_size = 0.8
test_size = 0.1
valid_size = 0.1
train_x,test_x,train_y,test_y = train_test_split(data,digits.target,train_size=train_size)
val_x,test_x,val_y,test_y = train_test_split(test_x,test_y,train_size=(valid_size/(valid_size + test_size)))


train_split_ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyperparameter_svm = [0.00001,0.0001,0.001,0.1,1,2,5,10,15,20,30]
collection = []
for train_ratio in train_split_ratio:
    model_collection = []
    if train_ratio != 1:
        X_train,_,Y_train,_ = create_train_test_split(X = train_x,Y = train_y,train_size=train_ratio)
    else:
        X_train,Y_train = train_x,train_y
    (unique,counts) = np.unique(Y_train,return_counts=True)
    freq = np.asarray((unique,counts)).T
    
    for gamma_val in hyperparameter_svm:
        
        val_metrics,model = g_SVM_metrics(train_X = X_train,train_Y=Y_train,test_X=val_x,test_Y = val_y,hyperparameter=gamma_val)
        if val_metrics['f1'] >= 0.4:
            if val_metrics:
                candidate = {
                    "% of training data used":train_ratio*100,
                    "Validation F1 Score":val_metrics['f1'],
                    "Validation Accuracy":val_metrics['acc'],
                    "Hyperparameter":gamma_val
                }
            model_collection.append(candidate)
            output_folder = "./models/"
            name = "train_size_{}_svm_gamma_{}.sav".format(train_ratio*100,gamma_val)
            pickle.dump(model, open(os.path.join(output_folder,name), 'wb'))
        else:
            print("Skipping for gamma ",gamma_val," and train size ",train_ratio," due to very low validation F1 Score i.e. ",val_metrics['f1'])

    max_valid_f1_score = max(model_collection,key=lambda x:x['Validation F1 Score'])
    best_model_folder = "train_size_{}_svm_gamma_{}.sav".format(max_valid_f1_score["% of training data used"],max_valid_f1_score["Hyperparameter"])
    path = os.path.join(output_folder,best_model_folder)
    best_model = pickle.load(open(path, 'rb'))
    test_metrics,prediction = g_test_metrics(best_model,test_x,test_y)
    print("\n")
    print("Test Metrics for ",max_valid_f1_score["% of training data used"]," ",max_valid_f1_score["Hyperparameter"]," ",test_metrics)
    plt.figure()
    cm = confusion_matrix(test_y,prediction)
    cm
    sns.heatmap(cm,annot=True)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    filename = "Confusion-Matrix for "+str(train_ratio*100)+" training data.png"
    plt.savefig(filename)
    del prediction
    if test_metrics:
        test_info = {
            "Test Accuracy":test_metrics['acc'],
            "Test F1 Score":test_metrics['f1']
        }
    max_valid_f1_score.update(test_info)
    collection.append(max_valid_f1_score)




df = pd.DataFrame(collection)

print(df)
plt.figure()
plt.plot(df['% of training data used'],df['Test F1 Score'])
plt.xlabel("Training data %")
plt.ylabel("Test F1 Score")
plt.title("Amount of Training data   Vs.  Test F1 Score")
plt.savefig("dataVSf1.png")