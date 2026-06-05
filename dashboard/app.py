import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import pickle
import time
from pathlib import Path

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide"
)

# ─────────────────────────────────────────────
# Bold medical theme: white, green, red
# ─────────────────────────────────────────────

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: #FFFFFF;
            color: #0A0A0A;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            padding: 2.5rem 3rem 3rem 3rem;
            max-width: 1100px;
        }

        .app-header {
            margin-bottom: 2.5rem;
        }

        .app-eyebrow {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #16A34A;
            margin-bottom: 0.5rem;
        }

        .app-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #0A0A0A;
            line-height: 1.1;
            letter-spacing: -1px;
            margin: 0 0 0.5rem 0;
        }

        .app-title span {
            color: #16A34A;
        }

        .app-subtitle {
            font-size: 0.9rem;
            color: #6B7280;
            font-weight: 400;
            max-width: 560px;
            line-height: 1.6;
        }

        .header-divider {
            border: none;
            border-top: 2px solid #0A0A0A;
            margin: 1.5rem 0 2rem 0;
        }

        .section-label {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #9CA3AF;
            margin-bottom: 1rem;
            margin-top: 2rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #F3F4F6;
        }

        .stat-card {
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
        }

        .stat-card.green { border-top: 3px solid #16A34A; }
        .stat-card.red   { border-top: 3px solid #DC2626; }
        .stat-card.dark  { border-top: 3px solid #0A0A0A; }
        .stat-card.blue  { border-top: 3px solid #2563EB; }

        .stat-label {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #9CA3AF;
            margin-bottom: 0.4rem;
        }

        .stat-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0A0A0A;
            line-height: 1;
        }

        .stat-sub {
            font-size: 0.75rem;
            color: #9CA3AF;
            margin-top: 0.3rem;
        }

        .input-description {
            font-size: 0.82rem;
            color: #6B7280;
            line-height: 1.6;
            margin-bottom: 1.2rem;
        }

        .feature-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.75rem;
            gap: 0.8rem;
        }

        .feature-name {
            font-size: 0.8rem;
            font-weight: 500;
            color: #374151;
            width: 210px;
            flex-shrink: 0;
        }

        .feature-bar-wrap {
            flex: 1;
            background: #F3F4F6;
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
        }

        .feature-bar-fill {
            height: 100%;
            border-radius: 4px;
            background: #16A34A;
        }

        .feature-bar-fill.negative {
            background: #DC2626;
        }

        .feature-coef {
            font-size: 0.75rem;
            font-family: 'DM Mono', monospace;
            color: #9CA3AF;
            width: 60px;
            text-align: right;
            flex-shrink: 0;
        }

        .result-box {
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin-top: 0.5rem;
        }

        .result-low      { background: #F0FDF4; border: 2px solid #16A34A; }
        .result-moderate { background: #FFFBEB; border: 2px solid #D97706; }
        .result-high     { background: #FEF2F2; border: 2px solid #DC2626; }

        .result-eyebrow {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .result-low      .result-eyebrow { color: #16A34A; }
        .result-moderate .result-eyebrow { color: #D97706; }
        .result-high     .result-eyebrow { color: #DC2626; }

        .result-title {
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .result-low      .result-title { color: #15803D; }
        .result-moderate .result-title { color: #B45309; }
        .result-high     .result-title { color: #B91C1C; }

        .result-probability {
            font-size: 0.85rem;
            color: #6B7280;
            margin-top: 0.3rem;
        }

        .result-recommendation {
            font-size: 0.82rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(0,0,0,0.08);
            color: #374151;
            line-height: 1.6;
        }

        .disclaimer {
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-left: 3px solid #D97706;
            border-radius: 0 8px 8px 0;
            padding: 0.9rem 1.1rem;
            font-size: 0.78rem;
            color: #6B7280;
            line-height: 1.6;
            margin-top: 2.5rem;
        }

        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #0A0A0A;
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.88rem;
            width: 100%;
        }

        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #1F2937;
        }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "model" / "diabetes_model.pkl"
METRICS_PATH = REPO_ROOT / "model" / "metrics.json"

RISK_LOW_MAX = 40       # diabetes probability (%) treated as low risk
RISK_MODERATE_MAX = 70  # upper bound of moderate risk

DEFAULT_METRICS = {
    "model_name": "Logistic Regression (balanced)",
    "accuracy_pct": 76.2,
    "roc_auc": 0.837,
    "total_patients": 768,
    "no_diabetes": 500,
    "has_diabetes": 268,
    "no_diabetes_pct": 65.1,
    "has_diabetes_pct": 34.9,
    "feature_importance": [],
}


@st.cache_resource
def load_model():
    """Load the trained pipeline, stopping with a clear error if absent."""
    try:
        with open(MODEL_PATH, "rb") as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        st.error(
            "Trained model not found. Run `python3 model/train.py` "
            "from the repository root, then reload this page."
        )
        st.stop()


@st.cache_data
def load_metrics():
    """Load metrics written by train.py, falling back to known defaults."""
    try:
        with open(METRICS_PATH) as metrics_file:
            return json.load(metrics_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_METRICS


pipeline = load_model()
metrics = load_metrics()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
    <div class="app-header">
        <div class="app-eyebrow">Clinical Decision Support Tool</div>
        <div class="app-title">Diabetes Risk <span>Predictor</span></div>
        <div class="app-subtitle">
            Enter patient clinical measurements to assess diabetes risk probability
            using a logistic regression model trained on 768 patients.
        </div>
    </div>
    <hr class="header-divider">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Dataset stats
# ─────────────────────────────────────────────

st.markdown('<div class="section-label">Dataset Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

def count_up(placeholder, target, color, suffix=""):
    """Animate the stat once per session, then render statically."""
    def render(value):
        placeholder.markdown(f"""
            <div class="stat-value" style="color:{color};">
                {value:,}{suffix}
            </div>
        """, unsafe_allow_html=True)

    if st.session_state.get("stats_animated"):
        render(target)
        return

    whole = int(round(target))
    steps = 30
    step_size = max(1, whole // steps)
    for i in range(0, whole + step_size, step_size):
        render(min(i, whole))
        time.sleep(0.018)
    render(target)

with c1:
    st.markdown('<div class="stat-card dark"><div class="stat-label">Total Patients</div>', unsafe_allow_html=True)
    p1 = st.empty()
    st.markdown('<div class="stat-sub">Pima Indians dataset</div></div>', unsafe_allow_html=True)
    count_up(p1, metrics["total_patients"], "#0A0A0A")

with c2:
    st.markdown('<div class="stat-card green"><div class="stat-label">No Diabetes</div>', unsafe_allow_html=True)
    p2 = st.empty()
    st.markdown(f'<div class="stat-sub">{metrics["no_diabetes_pct"]}% of dataset</div></div>', unsafe_allow_html=True)
    count_up(p2, metrics["no_diabetes"], "#16A34A")

with c3:
    st.markdown('<div class="stat-card red"><div class="stat-label">Has Diabetes</div>', unsafe_allow_html=True)
    p3 = st.empty()
    st.markdown(f'<div class="stat-sub">{metrics["has_diabetes_pct"]}% of dataset</div></div>', unsafe_allow_html=True)
    count_up(p3, metrics["has_diabetes"], "#DC2626")

with c4:
    st.markdown('<div class="stat-card blue"><div class="stat-label">Model Accuracy</div>', unsafe_allow_html=True)
    p4 = st.empty()
    st.markdown(f'<div class="stat-sub">{metrics.get("model_name", "Model")} | ROC-AUC {metrics.get("roc_auc", "n/a")}</div></div>', unsafe_allow_html=True)
    count_up(p4, metrics["accuracy_pct"], "#2563EB", suffix="%")

st.session_state.stats_animated = True

# ─────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────

st.markdown('<div class="section-label">Feature Importance: Scaled Coefficients</div>', unsafe_allow_html=True)

importance_rows = metrics.get("feature_importance", [])

feature_labels = {
    "Pregnancies"             : "Pregnancies",
    "Glucose"                 : "Blood Glucose",
    "BloodPressure"           : "Blood Pressure",
    "SkinThickness"           : "Skin Fold Thickness",
    "Insulin"                 : "Insulin Level",
    "BMI"                     : "Body Mass Index (BMI)",
    "DiabetesPedigreeFunction": "Genetic Risk Score (DPF)",
    "Age"                     : "Age"
}

if importance_rows:
    max_abs = max(abs(row["value"]) for row in importance_rows)
    for row in importance_rows:
        label     = feature_labels.get(row["feature"], row["feature"])
        value     = row["value"]
        bar_width = round((abs(value) / max_abs) * 100, 1)
        negative  = row.get("signed") and value < 0
        bar_class = "feature-bar-fill negative" if negative else "feature-bar-fill"
        coef_str  = f"+{value:.3f}" if row.get("signed") and value > 0 else f"{value:.3f}"

        st.markdown(f"""
            <div class="feature-row">
                <div class="feature-name">{label}</div>
                <div class="feature-bar-wrap">
                    <div class="{bar_class}" style="width:{bar_width}%;"></div>
                </div>
                <div class="feature-coef">{coef_str}</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("Run `python3 model/train.py` to generate feature importance.")

# ─────────────────────────────────────────────
# Patient input
# ─────────────────────────────────────────────

st.markdown('<div class="section-label">Patient Assessment</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="input-description">
        Enter the patient's clinical measurements below. All fields are required.
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies    = st.number_input("Number of Pregnancies",           min_value=0,   max_value=20,   value=1,    step=1)
    glucose        = st.number_input("Blood Glucose Level (mg/dL)",     min_value=0,   max_value=300,  value=110,  step=1)
    blood_pressure = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0,   max_value=200,  value=72,   step=1)
    skin_thickness = st.number_input("Skin Fold Thickness (mm)",        min_value=0,   max_value=100,  value=20,   step=1)

with col2:
    insulin = st.number_input("2-Hour Serum Insulin (μU/mL)",           min_value=0,   max_value=900,  value=80,   step=1)
    bmi     = st.number_input("Body Mass Index, BMI (kg/m²)",          min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
    dpf     = st.number_input("Genetic Risk Score (DPF)",               min_value=0.0, max_value=3.0,  value=0.35, step=0.01, format="%.3f")
    age     = st.number_input("Patient Age (years)",                    min_value=1,   max_value=120,  value=30,   step=1)

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

if st.button("Run Risk Assessment", type="primary"):

    patient = pd.DataFrame([{
        "Pregnancies"             : pregnancies,
        "Glucose"                 : glucose,
        "BloodPressure"           : blood_pressure,
        "SkinThickness"           : skin_thickness,
        "Insulin"                 : insulin,
        "BMI"                     : bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age"                     : age
    }])

    with st.spinner("Analysing patient data..."):
        prediction  = pipeline.predict(patient)[0]
        probability = pipeline.predict_proba(patient)[0]

    diabetes_prob    = round(probability[1] * 100, 1)
    no_diabetes_prob = round(probability[0] * 100, 1)

    st.markdown('<div class="section-label">Risk Assessment Result</div>', unsafe_allow_html=True)

    result_col, gauge_col = st.columns([1, 1])

    with gauge_col:
        # Gauge colour follows risk level
        if diabetes_prob <= RISK_LOW_MAX:
            gauge_color = "#16A34A"
        elif diabetes_prob <= RISK_MODERATE_MAX:
            gauge_color = "#D97706"
        else:
            gauge_color = "#DC2626"

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = diabetes_prob,
            number= {
                "suffix": "%",
                "font"  : {"size": 36, "color": "#0A0A0A", "family": "DM Sans"}
            },
            gauge = {
                "axis": {
                    "range"    : [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#E5E7EB",
                    "tickfont" : {"size": 11, "color": "#9CA3AF"}
                },
                "bar"      : {"color": gauge_color, "thickness": 0.25},
                "bgcolor"  : "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, RISK_LOW_MAX], "color": "#F0FDF4"},
                    {"range": [RISK_LOW_MAX, RISK_MODERATE_MAX], "color": "#FFFBEB"},
                    {"range": [RISK_MODERATE_MAX, 100], "color": "#FEF2F2"},
                ],
                "threshold": {
                    "line"     : {"color": "#0A0A0A", "width": 2},
                    "thickness": 0.75,
                    "value"    : diabetes_prob
                }
            },
            title = {
                "text" : "Diabetes Probability",
                "font" : {"size": 13, "color": "#6B7280", "family": "DM Sans"},
                "align": "center"
            }
        ))

        fig.update_layout(
            height        = 270,
            margin        = dict(t=40, b=10, l=30, r=30),
            paper_bgcolor = "white",
            font          = dict(family="DM Sans")
        )

        st.plotly_chart(fig, width="stretch")

    with result_col:
        if diabetes_prob <= RISK_LOW_MAX:
            box_class = "result-box result-low"
            title     = "Low Risk"
            recommend = "Current measurements do not indicate elevated diabetes risk. Recommend routine preventive care and annual screening."
        elif diabetes_prob <= RISK_MODERATE_MAX:
            box_class = "result-box result-moderate"
            title     = "Moderate Risk"
            recommend = "Elevated risk indicators detected. Recommend lifestyle intervention, dietary review, and follow-up testing within 3 months."
        else:
            box_class = "result-box result-high"
            title     = "High Risk"
            recommend = "Significant diabetes risk detected. Immediate referral for confirmatory diagnostic testing (HbA1c, fasting glucose) is advised."

        st.markdown(f"""
            <div class="{box_class}">
                <div class="result-eyebrow">Assessment Complete</div>
                <div class="result-title">{title}</div>
                <div class="result-probability">
                    Diabetes probability: <strong>{diabetes_prob}%</strong>
                    &nbsp;·&nbsp;
                    No diabetes: <strong>{no_diabetes_prob}%</strong>
                </div>
                <div class="result-recommendation">{recommend}</div>
            </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Disclaimer
# ─────────────────────────────────────────────

st.markdown("""
    <div class="disclaimer">
        <strong>Clinical Disclaimer:</strong> This tool is intended for educational
        and research purposes only. It is not a substitute for professional medical
        diagnosis. All predictions should be reviewed by a qualified healthcare
        professional before any clinical decision is made.
    </div>
""", unsafe_allow_html=True)