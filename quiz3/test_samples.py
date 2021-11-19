from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from utils import svm_load_predict, dtree_load_predict
import numpy as np

digits = load_digits()
data = digits.data

def test_digits_correct_0_svm():
    X = data[0]
    X = np.array(X).reshape(1, -1)
    pred, acc = svm_load_predict(X, 0)
    assert pred==0
    assert acc>0.75

def test_digits_correct_1_svm():
    X = data[1]
    X = np.array(X).reshape(1, -1)
    pred, acc = svm_load_predict(X, 1)
    assert pred==1
    assert acc>0.75

def test_digits_correct_2_svm():
    X = data[2]
    X = np.array(X).reshape(1, -1)
    pred, acc = svm_load_predict(X, 2)
    assert pred==2
    assert acc>0.75

def test_digits_correct_3_svm():
    X = data[3]
    X = np.array(X).reshape(1, -1)
    pred, acc = svm_load_predict(X, 3)
    assert pred==3
    assert acc>0.75

def test_digits_correct_4_svm():
    X = data[4]
    X = np.array(X).reshape(1, -1)
    pred, acc = svm_load_predict(X, 4)
    assert pred==4
    assert acc>0.75

def test_digits_correct_5_svm():
    X = data[5]
    X = np.array(X).reshape(1, -1)
    pred, acc = svm_load_predict(X, 5)
    assert pred==5
    assert acc>0.75

def test_digits_correct_6_svm():
    X = data[6]
    X = np.array(X).reshape(1, -1)
    pred, acc = svm_load_predict(X, 6)
    assert pred==6
    assert acc>0.75

def test_digits_correct_7_svm():
    X = data[7]
    X = np.array(X).reshape(1, -1)
    pred, acc = svm_load_predict(X, 7)
    assert pred==7
    assert acc>0.75

def test_digits_correct_8_svm():
    X = data[8]
    X = np.array(X).reshape(1, -1)
    pred, acc = svm_load_predict(X, 8)
    assert pred==8
    assert acc>0.75

def test_digits_correct_9_svm():
    X = data[9]
    X = np.array(X).reshape(1, -1)
    pred, acc = svm_load_predict(X, 9)
    assert pred==9
    assert acc>0.75

# DTREE

def test_digits_correct_0_dtree():
    X = data[0]
    X = np.array(X).reshape(1, -1)
    pred, acc = dtree_load_predict(X, 0)
    assert pred==0
    assert acc>0.75

def test_digits_correct_1_dtree():
    X = data[1]
    X = np.array(X).reshape(1, -1)
    pred, acc = dtree_load_predict(X, 1)
    assert pred==1
    assert acc>0.75

def test_digits_correct_2_dtree():
    X = data[2]
    X = np.array(X).reshape(1, -1)
    pred, acc = dtree_load_predict(X, 2)
    assert pred==2
    assert acc>0.75

def test_digits_correct_3_dtree():
    X = data[3]
    X = np.array(X).reshape(1, -1)
    pred, acc = dtree_load_predict(X, 3)
    assert pred==3
    assert acc>0.75

def test_digits_correct_4_dtree():
    X = data[4]
    X = np.array(X).reshape(1, -1)
    pred, acc = dtree_load_predict(X, 4)
    assert pred==4
    assert acc>0.75

def test_digits_correct_5_dtree():
    X = data[5]
    X = np.array(X).reshape(1, -1)
    pred, acc = dtree_load_predict(X, 5)
    assert pred==5
    assert acc>0.75

def test_digits_correct_6_dtree():
    X = data[6]
    X = np.array(X).reshape(1, -1)
    pred, acc = dtree_load_predict(X, 6)
    assert pred==6
    assert acc>0.75

def test_digits_correct_7_dtree():
    X = data[7]
    X = np.array(X).reshape(1, -1)
    pred, acc = dtree_load_predict(X, 7)
    assert pred==7
    assert acc>0.75

def test_digits_correct_8_dtree():
    X = data[8]
    X = np.array(X).reshape(1, -1)
    pred, acc = dtree_load_predict(X, 8)
    assert pred==8
    assert acc>0.75

def test_digits_correct_9_dtree():
    X = data[9]
    X = np.array(X).reshape(1, -1)
    pred, acc = dtree_load_predict(X, 9)
    assert pred==9
    assert acc>0.75