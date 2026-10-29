# Loan Approval Prediction using Machine Learning 🤖

## 📌 Project Overview

This project is a **Loan Approval Prediction web application** built using Machine Learning and Flask.

The application takes applicant information such as income, credit score, age, work experience, loan amount, existing debt, and employment status, and predicts whether the loan application is **Approved or Rejected**.

The Machine Learning model used in this project is **MLPClassifier** from Scikit-learn.

## 🌐 Web Application

The project includes a Flask-based web interface where users can enter their loan application details and get a prediction.

### Application Features

- Annual Income
- Credit Score
- Age
- Work Experience
- Loan Amount
- Existing Debt
- Employment Status
- Loan Approval Prediction

## 🧠 Machine Learning Workflow

The project follows these steps:

**Dataset → Train/Test Split → Feature Scaling → MLP Classifier → Prediction → Flask Web Application**

### Model

- MLPClassifier
- Activation: ReLU
- Hidden Layers: 16, 8
- Solver: Adam
- Max Iterations: 2000

### Preprocessing

- Train-Test Split
- StandardScaler

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLPClassifier
- StandardScaler
- Flask
- HTML
- CSS
- Jupyter Notebook / VS Code

## 📂 Project Structure

```text
Loan-Approval-Prediction/
│
├── app.py
├── train.py
├── train_model2.py
├── loan_approval_perceptron_full.csv
├── model.pkl
├── scaler.pkl
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## 🚀 How to Run

### 1. Install required libraries

```bash
pip install flask pandas numpy scikit-learn
```

### 2. Train the model

```bash
python train.py
```

This creates:

```text
model.pkl
scaler.pkl
```

### 3. Run Flask application

```bash
python app.py
```

### 4. Open in browser

```text
http://127.0.0.1:5000
```

## 🎯 Project Goal

The goal of this project is to demonstrate the complete Machine Learning workflow from **data preprocessing and model training to prediction through a Flask web application**.

## 👩‍💻 Author

**Roshani Salekar**

BSc IT | Data Science

<img width="1920" height="1080" alt="l" src="https://github.com/user-attachments/assets/af8278e7-a943-48d1-a609-cd6dc107bf57" />
<img width="1920" height="1080" alt="ll" src="https://github.com/user-attachments/assets/c6b9c36b-028f-4a71-b5e3-6d07917395ca" />

