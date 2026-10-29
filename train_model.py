import pandas as pd
import pickle

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("loan_approval_perceptron_full.csv")


# ==============================
# Features
# ==============================

features = [
    "Income",
    "Credit_Score",
    "Age",
    "Experience",
    "Loan_Amount",
    "Existing_Debt",
    "Employment_Status"
]

X = df[features]
y = df["Loan_Approved"]


# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# Scaling
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================
# Perceptron
# ==============================

model = Perceptron(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# ==============================
# Accuracy
# ==============================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print("Accuracy:", accuracy)


# ==============================
# Save Model
# ==============================

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


# ==============================
# Save Scaler
# ==============================

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


print("model.pkl created")
print("scaler.pkl created")