from flask import Flask, jsonify, request
import flask
import json
from ml_service import ML_Service
from invalidInputException import InvalidInputException

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'
	
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ml_service = ML_Service()
    request_data = json.loads(request.data)
    try:
        to_predict_list = request_data['data']
    except:
        raise InvalidInputException('\'data\' keyword is missing.')
    _,y_label = ml_service.predict(to_predict_list)
    return app.response_class(response=json.dumps({'prediction': y_label}),
                                  status=200,
                                  mimetype='application/json')

@app.route('/checkModel', methods=['POST'])
def checkPerformance():
    ml_service = ML_Service()
    request_data = json.loads(request.data)
    try:
        to_predict_list = request_data['data']
        y_list = request_data['y']
    except:
        raise InvalidInputException('\'data\' or \'Y\' keyword is missing.')
    
    result = ml_service.checkPerformance(to_predict_list, y_list)

    return app.response_class(response=json.dumps({'performance result': result}),
                                  status=200,
                                  mimetype='application/json')

@app.errorhandler(Exception)          
def basic_error(e):
    if isinstance(e,InvalidInputException):
        status_code = 400
    else:
        status_code = 500
    return app.response_class(response=json.dumps({'Error': str(e)}),
                                  status=status_code,
                                  mimetype='application/json')     
	

if __name__ == '__main__':
    app.run()	
	