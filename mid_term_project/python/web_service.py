from flask import Flask
import pickle
from flask import Flask
from flask import request
from flask import jsonify

app = Flask('ping') # give an identity to your web service

def load_file(filename: str):
    print(f'Filename: {filename}')
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

dv_home, model_home = load_file("../model/model_result_home.bin")
dv_away, model_away = load_file("../model/model_result_away.bin")

@app.route('/predict_home', methods=['POST']) 
def predict():
    game = request.get_json()
    
    X = dv_home.transform([game])
    get_home_win_draw_proba = round(model_home.predict_proba(X)[0, 1], 3)
    get_home_win_draw = get_home_win_draw_proba >= 0.5
    
    print(f'Get proba: {get_home_win_draw_proba}')
    
    result = {"get_home_win_draw_proba": float(get_home_win_draw_proba), 
              "get_home_win_draw": bool(get_home_win_draw)}
    
    return jsonify(result)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696
