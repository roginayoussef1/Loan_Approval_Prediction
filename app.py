from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    applicant_income = float(request.form['ApplicantIncome'])
    coapplicant_income = float(request.form['CoapplicantIncome'])
    loan_amount = float(request.form['LoanAmount'])
    loan_term = float(request.form['Loan_Amount_Term'])
    credit_history = float(request.form['Credit_History'])

    data = np.array([[
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history
    ]])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        result = 'Loan Approved ✅'
    else:
        result = 'Loan Rejected ❌'

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run()