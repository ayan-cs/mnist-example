from flask import Flask
from flask import request
from flask_restx import Resource, Api
from werkzeug.utils import cached_property
import numpy as np
import pickle

best_model_svm = '.mnist-code/models/best_svm.sav'
best_model_tree = '.mnist-code/models/best_dtree.sav'

app = Flask(__name__)
api = Api(app)


@app.route("/predict_svm",methods=['POST'])
def predict_svm():
    model = pickle.load(open(best_model_svm, 'rb'))
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1,-1)
    predict = model.predict(image)
    print("Image Sent from SVM :- ",image)
    print(str(predict[0]))
    return str(predict[0])

@app.route("/predict_tree",methods=['POST'])
def predict_tree():
    model = pickle.load(open(best_model_tree, 'rb'))
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1,-1)
    predict = model.predict(image)
    print("Image Sent from tree:- ",image)
    print(str(predict[0]))
    return str(predict[0])

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000)