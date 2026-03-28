import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os

# Load dataset
data = pd.read_csv("data/diabetes.csv")

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(data.head())
print(f"\nShape: {data.shape}")
print("\nMissing values:")
print(data.isnull().sum())

# Class distribution
outcome_counts = data["Outcome"].value_counts()
no_diabetes  = outcome_counts[0]
has_diabetes = outcome_counts[1]
total        = len(data)

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)
print(f"No Diabetes  : {no_diabetes} ({no_diabetes / total * 100:.1f}%)")
print(f"Has Diabetes : {has_diabetes} ({has_diabetes / total * 100:.1f}%)")
print(f"Ratio        : {no_diabetes / has_diabetes:.2f}:1")

# Features and target
X = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
          "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = data["Outcome"]

# Stratified 70/30 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set : {X_train.shape[0]} patients")
print(f"Testing set  : {X_test.shape[0]} patients")

# Pipeline: StandardScaler normalises all features to the same scale
# so coefficients can be compared fairly across features
# This ensures Glucose, BMI and DPF are judged on equal terms
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(random_state=42, max_iter=1000))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred   = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Has Diabetes"]))

# Feature importance using scaled coefficients
# Now fair to compare directly across all features
model      = pipeline.named_steps["model"]
scaler     = pipeline.named_steps["scaler"]

coefficients = pd.DataFrame({
    "Feature"    : X.columns,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False)

print("\nFeature importance (scaled — comparable across all features):")
for _, row in coefficients.iterrows():
    direction = "increases" if row["Coefficient"] > 0 else "decreases"
    print(f"  {row['Feature']}: {row['Coefficient']:.4f} ({direction} risk)")

# Save the full pipeline so the dashboard uses the same scaler
os.makedirs("model", exist_ok=True)
with open("model/diabetes_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModel saved to: model/diabetes_model.pkl")