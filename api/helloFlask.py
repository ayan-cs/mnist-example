from flask import Flask
from flask import request
from flask_restx import Resource, Api
from werkzeug.utils import cached_property
import numpy as np
import pickle


best_model = '../mnist-code/models/model-3.0-10-0.7483.sav'

app = Flask(__name__)
api = Api(app)

#@app.route("/")
#def hello_world():
#    return "<p> Hello World!</p>"
@api.route('/hello')
class HelloWorld(Resource):
        def get(self):
            return {'hello': 'world'}


@app.route("/predict",methods=['POST'])
def predict():
    model = pickle.load(open(best_model, 'rb'))
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1,-1)
    predict = model.predict(image)
    print("Iamge Sent :- ",image)
    print(str(predict[0]))
    return str(predict[0])

if __name__=='__main__':
    #app.run(debug=True)
    app.run()
