import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('/Users/moon/Desktop/Batch862/diabetes_project/file/diabetes.csv')
print("="*70)
print("\n___DATASET___")
print("="*70)
print(data.head())
print(f"\nShape: {data.shape}")

print("\n___DATASET INFO___")
print(data.info())

print("\n___DESCRIPTIVE STATISTICS___")
print(data.describe())

print("\n___MISSING VALUES___")
print(data.isnull().sum())

# QUESTION 3: Analyze class distribution
print("\n" + "="*70)
print("QUESTION 3: CLASS IMBALANCE")
print("="*70)

outcome_counts = data['Outcome'].value_counts()
print("\nClass Distribution:")
print(outcome_counts)

no_diabetes = outcome_counts[0]
has_diabetes = outcome_counts[1]
total = len(data)

print(f"\nNo Diabetes: {no_diabetes} ({no_diabetes/total*100:.1f}%)")
print(f"Has Diabetes: {has_diabetes} ({has_diabetes/total*100:.1f}%)")
print(f"Ratio: {no_diabetes/has_diabetes:.2f}:1")

print("\nSOLUTION:")
print("How imbalance is handled:")
print("  1. Keep original distribution (no resampling)")
print("  2. Use stratified split to maintain proportions")
print("  3. Logistic regression handles imbalance naturally through probabilities")
print("  4. Evaluate both classes separately using classification report")

# Plot class distribution
plt.bar(['No Diabetes', 'Has Diabetes'], outcome_counts.values, color=['blue', 'red'])
plt.title('Class Distribution')
plt.ylabel('Number of Patients')
for i, v in enumerate(outcome_counts.values):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.show()

# Prepare features and target
print("\n" + "="*70)
print("DATA PREPARATION")
print("="*70)

X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']

print("\nFeatures (X):")
print(X.columns.tolist())
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining: {X_train.shape[0]} patients")
print(f"Testing: {X_test.shape[0]} patients")

print("\nTraining Set Distribution:")
print(y_train.value_counts())

print("\nTesting Set Distribution:")
print(y_test.value_counts())

# QUESTION 1: Build model
print("\n" + "="*70)
print("QUESTION 1: BUILD MODEL")
print("="*70)

model = LogisticRegression(random_state=42, max_iter=1000)

print("\nTraining model...")
model.fit(X_train, y_train)
print("Model trained successfully")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n___MODEL PERFORMANCE___")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n___CLASSIFICATION REPORT___")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Has Diabetes']))

print("\nSOLUTION:")
print(f"  Logistic Regression model created successfully")
print(f"  Model achieves {accuracy*100:.2f}% accuracy on test data")
print(f"  Model can predict diabetes risk to help doctors flag at-risk patients")
print(f"  Trained on {X_train.shape[0]} patients, tested on {X_test.shape[0]} patients")

# QUESTION 2: Feature importance
print("\n" + "="*70)
print("QUESTION 2: FEATURE IMPORTANCE")
print("="*70)

print("\n___MODEL COEFFICIENTS___")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_[0])

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

coefficients['Abs_Coef'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values('Abs_Coef', ascending=False)

print("\n___COEFFICIENTS (SORTED BY IMPORTANCE)___")
print(coefficients[['Feature', 'Coefficient']])

print("\nSOLUTION:")
print("Most significant predictors of diabetes:")
for i in range(3):
    feat = coefficients.iloc[i]['Feature']
    coef = coefficients.iloc[i]['Coefficient']
    effect = "increases" if coef > 0 else "decreases"
    print(f"  {i+1}. {feat}: {coef:.4f} ({effect} diabetes risk)")

# Plot coefficients
plt.barh(coefficients['Feature'], coefficients['Coefficient'], 
         color=['red' if x > 0 else 'blue' for x in coefficients['Coefficient']])
plt.xlabel('Coefficient')
plt.title('Feature Importance')
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

# Test on sample patients
print("\n" + "="*70)
print("TEST MODEL PREDICTIONS")
print("="*70)

patient1 = pd.DataFrame({
    'Pregnancies': [6],
    'Glucose': [148],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [0],
    'BMI': [33.6],
    'DiabetesPedigreeFunction': [0.627],
    'Age': [50]
})

patient2 = pd.DataFrame({
    'Pregnancies': [1],
    'Glucose': [85],
    'BloodPressure': [66],
    'SkinThickness': [29],
    'Insulin': [0],
    'BMI': [26.6],
    'DiabetesPedigreeFunction': [0.351],
    'Age': [31]
})

print("\n___PATIENT 1 (HIGH RISK PROFILE)___")
print(f"Age: {patient1['Age'][0]}, Glucose: {patient1['Glucose'][0]}, BMI: {patient1['BMI'][0]}")
pred1 = model.predict(patient1)[0]
prob1 = model.predict_proba(patient1)[0]
print("Predicted class:", pred1)
print("Predicted probability:", prob1)

print("\n___PATIENT 2 (LOW RISK PROFILE)___")
print(f"Age: {patient2['Age'][0]}, Glucose: {patient2['Glucose'][0]}, BMI: {patient2['BMI'][0]}")
pred2 = model.predict(patient2)[0]
prob2 = model.predict_proba(patient2)[0]
print("Predicted class:", pred2)
print("Predicted probability:", prob2)

# Interactive prediction for user input
print("\n" + "="*70)
print("PREDICT FOR NEW PATIENT")
print("="*70)

print("\nEnter patient details:")
pregnancies = int(input("Number of Pregnancies: "))
glucose = float(input("Glucose Level: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

new_patient = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree],
    'Age': [age]
})

print("\n___PREDICTION RESULT___")
prediction = model.predict(new_patient)[0]
probability = model.predict_proba(new_patient)[0]

print(f"Predicted class: {prediction}")
print(f"Predicted probability: {probability}")
print(f"\nResult: {'HAS DIABETES' if prediction == 1 else 'NO DIABETES'}")
print(f"Diabetes probability: {probability[1]:.3f} ({probability[1]*100:.1f}%)")
print(f"No diabetes probability: {probability[0]:.3f} ({probability[0]*100:.1f}%)")

if probability[1] > 0.7:
    print("\nRECOMMENDATION: HIGH RISK - Immediate diagnostic testing advised")
elif probability[1] > 0.4:
    print("\nRECOMMENDATION: MODERATE RISK - Close monitoring recommended")
else:
    print("\nRECOMMENDATION: LOW RISK - Regular preventive care sufficient")

# Summary
print("\n" + "="*70)
print("PROJECT SUMMARY")
print("="*70)

print("\n___QUESTION 1: MODEL FOR EARLY RISK DETECTION___")
print(f"ANSWER: Logistic Regression model created with {accuracy*100:.2f}% accuracy")
print(f"        Helps doctors flag at-risk patients by providing risk probabilities")

print("\n___QUESTION 2: MOST SIGNIFICANT PREDICTORS___")
print("ANSWER: Top 3 most significant features:")
for i in range(3):
    print(f"        {i+1}. {coefficients.iloc[i]['Feature']}: {coefficients.iloc[i]['Coefficient']:.4f}")

print(f"\n___QUESTION 3: CLASS IMBALANCE HANDLING___")
print(f"ANSWER: Dataset has {no_diabetes/has_diabetes:.2f}:1 imbalance")
print(f"        Handled by: stratified split, natural distribution, separate class evaluation")
