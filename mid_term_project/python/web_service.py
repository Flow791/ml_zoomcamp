from flask import Flask
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

app = Flask('ping') # give an identity to your web service

def load_file(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

dv, model = load_file("model_result.bin")

@app.route('/predict', methods=['POST']) 
def predict():
    game = request.get_json()
    
    X = dv.transform([game])
    get_result_proba = np.round(model.predict_proba(X), 3)[0]
    
    print(f'Get proba: {get_result_proba}')
    
    result = {"home_win_proba": float(get_result_proba[1]), 
              "away_win_proba": float(get_result_proba[2]),
              "draw_proba": float(get_result_proba[0])}
    
    return jsonify(result)

if __name__ == '__main__':
    print('------> Run app <-------')
    app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696
