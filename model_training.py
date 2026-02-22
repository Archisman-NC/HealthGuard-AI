import pandas as pd 
import numpy as np 
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report

df = pd.read_csv('data/diabetes.csv')

zero_missing_cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

for col in zero_missing_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop("Outcome",axis=1)
y = df["Outcome"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled,y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Accuracy:", (accuracy_score(y_test, y_pred)))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n",classification_report(y_test,y_pred))
