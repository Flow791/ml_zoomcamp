from flask import Flask
import pickle
from flask import Flask
from flask import request
from flask import jsonify

app = Flask('ping') # give an identity to your web service

def load_file(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
dv = load_file("../model/dv.bin")
model = load_file("../model/model.bin")

@app.route('/predict', methods=['POST']) # use decorator to add Flask's functionality to our function
def predict():
    client = request.get_json()
    
    print('Get client')

    X = dv.transform([client])
    get_subscription_proba = round(model.predict_proba(X)[0, 1], 3)
    get_subscription = get_subscription_proba >= 0.5
    
    print(f'Get proba: {get_subscription_proba}')
    
    result = {"get_subscription_proba": float(get_subscription_proba), 
              "get_subscription": bool(get_subscription)}
    
    return jsonify(result)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696
