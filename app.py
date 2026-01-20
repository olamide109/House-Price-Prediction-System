from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/house_price_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        int(request.form['overallqual']),
        float(request.form['grlivarea']),
        float(request.form['totalbsmtsf']),
        int(request.form['garagecars']),
        int(request.form['fullbath']),
        int(request.form['yearbuilt'])
    ]

    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]

    return render_template(
        'index.html',
        prediction=f"₦{prediction:,.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True)
