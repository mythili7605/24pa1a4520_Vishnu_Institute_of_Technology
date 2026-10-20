# -*- coding: utf-8 -*-
"""
app.py – Flask backend for the Hypertension Digital Twin v2
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify

import database as db
from twin_engine import analyse_trend, generate_alerts, seed_demo_readings, classify_bp

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
MODEL_PATH = Path(__file__).parent / "rf_hypertension_model.pkl"
model = joblib.load(MODEL_PATH)
print(f"[OK] Model loaded: {MODEL_PATH}")

# Feature metadata
NUMERIC_FEATURES     = ["Age", "BMI", "Waist_cm", "Height_cm",
                         "Waist_to_Height", "PovertyRatio"]
CATEGORICAL_FEATURES = ["Sex", "Ethnicity", "Education",
                         "Diabetes", "BP_Medication"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

rf_clf    = model.named_steps["classifier"]
ohe       = (model.named_steps["preprocessor"]
             .named_transformers_["cat"].named_steps["onehot"])
cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
imp_all   = rf_clf.feature_importances_


def agg_importances():
    agg = {}
    for fn, imp in zip(NUMERIC_FEATURES, imp_all[:len(NUMERIC_FEATURES)]):
        agg[fn] = float(imp)
    for fn, imp in zip(cat_names, imp_all[len(NUMERIC_FEATURES):]):
        for cf in CATEGORICAL_FEATURES:
            if fn.startswith(cf + "_") or fn == cf:
                agg[cf] = agg.get(cf, 0.0) + float(imp)
                break
    return dict(sorted(agg.items(), key=lambda x: x[1], reverse=True))


AGG_IMP = agg_importances()


# ── RF prediction helper ──────────────────────────────────────────────────────
# Defined FIRST so it can be used during DB seeding below
def _predict_risk(p: dict):
    waist  = float(p.get("waist", 0) or 0)
    height = float(p.get("height", 165) or 165)
    bmi    = float(p.get("bmi", 25) or 25)
    whr    = waist / height if height > 0 else 0.5

    row = pd.DataFrame([{
        "Age":             float(p.get("age", 45)),
        "BMI":             bmi,
        "Waist_cm":        waist,
        "Height_cm":       height,
        "Waist_to_Height": whr,
        "PovertyRatio":    float(p.get("poverty_ratio", 2.5)),
        "Sex":             p.get("sex", "Male"),
        "Ethnicity":       str(p.get("ethnicity", "3.0")),
        "Education":       str(p.get("education", "3.0")),
        "Diabetes":        p.get("diabetes", "No"),
        "BP_Medication":   p.get("bp_medication", "No"),
    }])

    prob = float(model.predict_proba(row)[0][1]) * 100
    # This is a model estimate for the elevated-BP target,
    # not a clinical severity classification.
    if prob < 30:
        lvl = "Lower predicted likelihood"
    elif prob < 55:
        lvl = "Moderate predicted likelihood"
    else:
        lvl = "Higher predicted likelihood"
    return round(prob, 1), lvl


# ── DB init + seed ────────────────────────────────────────────────────────────
db.init_db()

if not db.patient_exists():
    print("[SEED] Creating demo Patient A ...")

    patient_profile = {
        "name": "Patient A (Demo)", "age": 45, "sex": "Male",
        "bmi": 29.0, "waist_cm": 98.0, "height_cm": 175.0,
        "ethnicity": "3.0", "education": "4.0", "poverty_ratio": 2.8,
        "diabetes": "No", "bp_medication": "No",
    }
    pid = db.create_patient(patient_profile)

    seed_params = {
        "age": 45, "bmi": 29.0, "waist": 98.0, "height": 175.0,
        "poverty_ratio": 2.8, "sex": "Male", "ethnicity": "3.0",
        "education": "4.0", "diabetes": "No", "bp_medication": "No",
    }

    for reading in seed_demo_readings():
        risk_prob, risk_lvl = _predict_risk(seed_params)
        reading["risk_score"] = risk_prob
        reading["risk_level"] = risk_lvl
        db.add_reading(pid, reading)

    # Fire initial alerts based on the seeded readings
    readings = db.get_readings(pid)
    last = readings[-1]
    generate_alerts(pid, readings,
                    last["systolic"], last["diastolic"],
                    last["risk_score"], last["risk_level"])
    print(f"[SEED] Patient A seeded with {len(readings)} readings.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─ Patients ──────────────────────────────────────────────────────────────────

@app.route("/api/patients", methods=["GET"])
def list_patients():
    return jsonify(db.list_patients())


@app.route("/api/patients", methods=["POST"])
def create_patient():
    data = request.get_json()
    pid  = db.create_patient(data)
    return jsonify({"id": pid, "message": "Patient created"})


@app.route("/api/patients/<int:pid>", methods=["GET"])
def get_patient(pid):
    patient = db.get_patient(pid)
    if not patient:
        return jsonify({"error": "Not found"}), 404

    readings = db.get_readings(pid)
    trend    = analyse_trend(readings)
    alerts   = db.get_alerts(pid, unack_only=True)

    last = readings[-1] if readings else None
    bp_label, _ = (classify_bp(last["systolic"], last["diastolic"])
                   if last else ("N/A", False))

    return jsonify({
        "patient":       patient,
        "readings":      readings,
        "trend":         trend,
        "alerts":        alerts,
        "alert_count":   len(alerts),
        "last_bp_label": bp_label,
    })


# ─ Readings ──────────────────────────────────────────────────────────────────

@app.route("/api/patients/<int:pid>/readings", methods=["POST"])
def add_reading(pid):
    patient = db.get_patient(pid)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    data = request.get_json()
    sbp  = float(data["systolic"])
    dbp  = float(data["diastolic"])
    ts   = data.get("timestamp", datetime.now().isoformat(timespec="seconds"))

    risk_prob, risk_lvl = _predict_risk({
        "age":           patient["age"],
        "bmi":           patient["bmi"],
        "waist":         patient["waist_cm"],
        "height":        patient["height_cm"],
        "poverty_ratio": patient["poverty_ratio"],
        "sex":           patient["sex"],
        "ethnicity":     patient["ethnicity"],
        "education":     patient["education"],
        "diabetes":      patient["diabetes"],
        "bp_medication": patient["bp_medication"],
    })

    reading_row = {
        "systolic":   sbp,
        "diastolic":  dbp,
        "heart_rate": data.get("heart_rate"),
        "risk_score": risk_prob,
        "risk_level": risk_lvl,
        "notes":      data.get("notes", ""),
        "timestamp":  ts,
    }
    db.add_reading(pid, reading_row)

    readings   = db.get_readings(pid)
    trend      = analyse_trend(readings)
    # Pass all readings including the latest one so the persistent-elevation
    # check can see the newest reading in its consecutive-reading window.
    new_alerts = generate_alerts(pid, readings, sbp, dbp, risk_prob, risk_lvl)
    bp_label, _ = classify_bp(sbp, dbp)

    return jsonify({
        "reading":    reading_row,
        "risk_score": risk_prob,
        "risk_level": risk_lvl,
        "trend":      trend,
        "bp_label":   bp_label,
        "new_alerts": new_alerts,
    })


# ─ Trend & Alerts ────────────────────────────────────────────────────────────

@app.route("/api/patients/<int:pid>/trend", methods=["GET"])
def get_trend(pid):
    readings = db.get_readings(pid)
    return jsonify(analyse_trend(readings))


@app.route("/api/patients/<int:pid>/alerts", methods=["GET"])
def get_alerts_route(pid):
    unack = request.args.get("unack", "true").lower() == "true"
    return jsonify(db.get_alerts(pid, unack_only=unack))


@app.route("/api/alerts/<int:aid>/ack", methods=["POST"])
def ack_alert(aid):
    db.acknowledge_alert(aid)
    return jsonify({"acknowledged": True})


# ─ Model info ────────────────────────────────────────────────────────────────

@app.route("/api/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type":          "Random Forest Classifier",
        "n_estimators":        rf_clf.n_estimators,
        "features":            ALL_FEATURES,
        "feature_importances": AGG_IMP,
        "accuracy":  0.6561,
        "roc_auc":   0.7225,
        "f1_score":  0.6359,
        "recall":    0.7067,
        "precision": 0.5779,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
