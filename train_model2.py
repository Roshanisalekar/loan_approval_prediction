import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load dataset
df = pd.read_csv("loan_approval_perceptron_full.csv")


# 2. Input features
X = df[
    [
        "Income",
        "Credit_Score",
        "Age",
        "Experience",
        "Loan_Amount",
        "Existing_Debt",
        "Employment_Status"
    ]
]


# 3. Target
y = df["Loan_Approved"]


# 4. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 5. Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 6. MLP Model
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42
)


# 7. Training
model.fit(X_train_scaled, y_train)


# 8. Prediction
y_pred = model.predict(X_test_scaled)


# 9. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("================================")
print("MLP LOAN APPROVAL MODEL")
print("================================")

print("Accuracy:", accuracy)


# 10. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 11. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 12. Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


# 13. Save scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


print("\nmodel.pkl created successfully!")
print("scaler.pkl created successfully!")