from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
app = Flask(__name__)
app.debug = True

# Load the trained model
post_result = {}
model=load("./trained_model.pkl")


@app.route('/')
def index():
    return "welcome weather api"
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    RH2M_value = data.get('RH2M')
    T2MDEW_value = data.get('T2MDEW')
    TS_value = data.get('TS')
    T2M_MAX_value = data.get('T2M_MAX')
    T2M_MIN_value = data.get('T2M_MIN')
# Tạo DataFrame với các giá trị đầu vào
    X_test = pd.DataFrame([[RH2M_value, T2MDEW_value, TS_value, T2M_MAX_value, T2M_MIN_value]],
                         columns=["RH2M", "T2MDEW", "TS", "T2M_MAX", "T2M_MIN"])
    y_pred = model.predict(X_test)
    post_result.update({'T2M': y_pred[0,0], 'PRECTOTCORR': y_pred[0,1]})
    return "Đã dự đoán"
@app.route('/api/weather')
def home():
   global post_result 
   return jsonify(post_result)

if __name__ == '__main__':
    app.run()