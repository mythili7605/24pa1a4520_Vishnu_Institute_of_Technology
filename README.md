# Hypertension Digital Twin

An interactive Digital Twin prototype for tracking blood pressure trajectories, calculating longitudinal trend slopes, screening according to ACC/AHA 2017 blood pressure guidelines, and estimating the statistical likelihood of Stage 1+ elevated blood pressure using a Random Forest classifier trained on NHANES adult data.

---

## Clinical Disclaimer

This project is an educational proof of concept for blood pressure monitoring and elevated-BP prediction. Model probabilities are statistical estimates based on the model's training data and input features; they are not diagnoses or validated estimates of an individual's clinical risk. Blood pressure categories and alerts are intended for screening and demonstration purposes only. The system has not been clinically validated and must not replace professional medical assessment.

---

## Features

- **ACC/AHA 2017 BP Screening**: Automatic classification of blood pressure readings into Normal BP, Elevated BP, Stage 1 Hypertension Range, Stage 2 Hypertension Range, and Hypertensive Crisis Range.
- **Trend Analysis**: Moving-window linear regression estimating SBP and DBP trajectory slopes (mmHg/day) over recent readings, with robust handling for duplicate or irregular timestamps.
- **Multi-rule Alert System**:
  - Hypertensive crisis range alerts with urgent medical evaluation guidance.
  - Stage 1 and Stage 2 screening alerts.
  - Persistent elevation alerts across consecutive readings.
  - Rising trend alerts.
  - Patient profile risk shift alerts.
- **Machine Learning Integration**: Random Forest classifier trained on 8,024 NHANES adult records without BP feature leakage, evaluating demographic and anthropometric factors (Age, BMI, Waist, Height, Poverty Ratio, Sex, Ethnicity, Education, Diabetes, BP Medication).
- **Interactive Web Dashboard**: Flask backend with Chart.js visualizations, real-time alert center, patient management, and risk likelihood gauge.

---

## Project Structure

```
├── app.py                     # Flask web server and API routes
├── database.py                # SQLite database interface & patient/reading CRUD
├── twin_engine.py             # BP classification, trend estimation & alert rules
├── train_model.py             # Random Forest pipeline training script (NHANES data)
├── test_digital_twin.py       # End-to-end validation suite (41 checks)
├── test_patient.py            # Single-patient profile prediction checks
├── rf_hypertension_model.pkl  # Trained scikit-learn model artifact
├── evaluation_report.txt      # Model performance metrics on hold-out set
├── test_report.txt            # Test execution report
├── templates/
│   └── index.html             # Dashboard frontend template
└── static/
    ├── css/dashboard.css      # Dashboard styling
    └── js/dashboard.js        # Dashboard state & charts
```

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

### 3. Run Validation Tests
```bash
python test_patient.py
python test_digital_twin.py
```
