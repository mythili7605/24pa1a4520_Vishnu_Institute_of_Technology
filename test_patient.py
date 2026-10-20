# -*- coding: utf-8 -*-
"""
test_patient.py – Quick single-patient prediction test
Usage: python test_patient.py
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "rf_hypertension_model.pkl"

print("=" * 60)
print("  SINGLE PATIENT PREDICTION TEST")
print("=" * 60)

model = joblib.load(MODEL_PATH)
print(f"[OK] Model loaded from {MODEL_PATH}\n")

# ── Test patients ─────────────────────────────────────────────
patients = [
    {
        "label":       "Young healthy male",
        "Age": 28, "BMI": 22.0, "Waist_cm": 75.0, "Height_cm": 178.0,
        "Waist_to_Height": 75/178, "PovertyRatio": 3.5,
        "Sex": "Male", "Ethnicity": "3.0", "Education": "5.0",
        "Diabetes": "No", "BP_Medication": "No",
    },
    {
        "label":       "Middle-aged overweight female",
        "Age": 52, "BMI": 31.5, "Waist_cm": 100.0, "Height_cm": 162.0,
        "Waist_to_Height": 100/162, "PovertyRatio": 1.8,
        "Sex": "Female", "Ethnicity": "4.0", "Education": "3.0",
        "Diabetes": "Borderline", "BP_Medication": "No",
    },
    {
        "label":       "Patient A profile (challenge demo)",
        "Age": 45, "BMI": 29.0, "Waist_cm": 98.0, "Height_cm": 175.0,
        "Waist_to_Height": 98/175, "PovertyRatio": 2.8,
        "Sex": "Male", "Ethnicity": "3.0", "Education": "4.0",
        "Diabetes": "No", "BP_Medication": "No",
    },
    {
        "label":       "Elderly diabetic on BP medication",
        "Age": 71, "BMI": 27.8, "Waist_cm": 96.0, "Height_cm": 168.0,
        "Waist_to_Height": 96/168, "PovertyRatio": 1.2,
        "Sex": "Male", "Ethnicity": "1.0", "Education": "2.0",
        "Diabetes": "Yes", "BP_Medication": "Yes",
    },
]

FEATURES = ["Age", "BMI", "Waist_cm", "Height_cm", "Waist_to_Height",
            "PovertyRatio", "Sex", "Ethnicity", "Education",
            "Diabetes", "BP_Medication"]

sep = "-" * 60
for p in patients:
    row = pd.DataFrame([{k: p[k] for k in FEATURES}])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]

    label_str  = "ELEVATED" if pred == 1 else "Normal"
    hint = ">>> Higher predicted likelihood of elevated BP <<<" if prob >= 0.55 else ""

    print(f"\nPatient : {p['label']}")
    print(f"  Age {p['Age']} yrs | BMI {p['BMI']} | Waist {p['Waist_cm']} cm")
    print(f"  Diabetes: {p['Diabetes']} | BP Med: {p['BP_Medication']}")
    print(f"  Predicted class  : {pred}  ({label_str})")
    print(f"  Predicted elevated-BP likelihood: {prob:.1%}  {hint}")
    print(sep)

print("\n[NOTE] These are software test outputs. Not clinically validated predictions.")
print("=" * 60)
