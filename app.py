from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)


# ==============================
# Load Model
# ==============================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# ==============================
# Load Scaler
# ==============================

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ==============================
# Home Page
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# Prediction
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        income = float(request.form["income"])
        credit_score = float(request.form["credit_score"])
        age = float(request.form["age"])
        experience = float(request.form["experience"])
        loan_amount = float(request.form["loan_amount"])
        existing_debt = float(request.form["existing_debt"])
        employment_status = int(request.form["employment_status"])


        # Create input
        data = np.array([[
            income,
            credit_score,
            age,
            experience,
            loan_amount,
            existing_debt,
            employment_status
        ]])


        # Scale input
        data_scaled = scaler.transform(data)


        # Prediction
        prediction = model.predict(data_scaled)[0]


        if prediction == 1:

            result = "Loan Approved"
            status = "approved"

        else:

            result = "Loan Rejected"
            status = "rejected"


        return render_template(
            "index.html",
            result=result,
            status=status
        )


    except Exception as e:

        return render_template(
            "index.html",
            result="Invalid input. Please check your values.",
            status="error"
        )


# ==============================
# Run Flask
# ==============================

if __name__ == "__main__":
    app.run(debug=True)