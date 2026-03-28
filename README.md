# Diabetes Risk Predictor

A clinical decision support tool that predicts diabetes risk probability
from patient measurements using logistic regression trained on 768 patients.

## Dataset

**Source**: Pima Indians Diabetes Database
**Patients**: 768 | **Features**: 8 clinical measurements | **Target**: Diabetic / Non-Diabetic

| Feature                  | Description                          |
| ------------------------ | ------------------------------------ |
| Pregnancies              | Number of pregnancies                |
| Glucose                  | Plasma glucose concentration (mg/dL) |
| BloodPressure            | Diastolic blood pressure (mmHg)      |
| SkinThickness            | Triceps skin fold thickness (mm)     |
| Insulin                  | 2-Hour serum insulin (μU/mL)         |
| BMI                      | Body mass index (kg/m²)              |
| DiabetesPedigreeFunction | Genetic diabetes likelihood score    |
| Age                      | Patient age (years)                  |

## Model

- Algorithm: Logistic Regression with StandardScaler
- Train/Test Split: 70% / 30% stratified
- Accuracy: 77%
- Top predictors: Glucose, BMI, Pregnancies

Features are scaled before training so coefficients are
comparable across all features regardless of their natural range.

## Class Distribution

- No Diabetes: 500 patients (65.1%)
- Has Diabetes: 268 patients (34.9%)
- Ratio: 1.87:1 — handled via stratified sampling and class-specific evaluation

## Project Structure

```
diabetes_project/
├── data/
│   └── diabetes.csv
├── model/
│   ├── train.py
│   └── diabetes_model.pkl
├── dashboard/
│   └── app.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

## Setup

```bash
pip3 install pandas scikit-learn matplotlib streamlit plotly
python3 model/train.py
streamlit run dashboard/app.py
```

## Dashboard Features

- Dataset overview with animated stat cards
- Feature importance chart with scaled coefficients
- Patient input form with 8 clinical fields
- Gauge chart showing diabetes probability
- Colour-coded risk result — Low / Moderate / High
- Clinical recommendation per risk level
- Medical disclaimer

## Authors

Onyebuchi Oluebube Emmanuela & Eniola Solomon Talabi
October 2025
