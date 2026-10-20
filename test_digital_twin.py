# -*- coding: utf-8 -*-
"""
test_digital_twin.py
======================
Comprehensive validation for the Hypertension Digital Twin PoC.

Covers:
  [A] Model validation    – metrics, baseline comparison, cross-validation, leakage check
  [B] Digital Twin Tests  – Test A (initial reading), Test B (update + alert),
                            Test C (idempotency, timestamps, rolling avg, missing values)
  [C] Full report         – saved to test_report.txt

Usage: python test_digital_twin.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

import database as db
import twin_engine as te

# ── Config ─────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent
MODEL_PATH = DATA_DIR / "rf_hypertension_model.pkl"
REPORT_PATH = DATA_DIR / "test_report.txt"
RANDOM_STATE = 42

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
SEP  = "=" * 65
SEP2 = "-" * 65

results = []   # (test_name, passed, detail)

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    results.append((name, cond, detail))
    print(f"  {status} {name}")
    if detail:
        print(f"         {detail}")
    return cond

print(SEP)
print("  HYPERTENSION DIGITAL TWIN -- FULL VALIDATION SUITE")
print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(SEP)


# ══════════════════════════════════════════════════════════════
# SECTION A -- MODEL VALIDATION
# ══════════════════════════════════════════════════════════════
print("\n[A] MODEL VALIDATION\n" + SEP2)

# A1. Load model
try:
    model = joblib.load(MODEL_PATH)
    check("A1: Model file loads without error", True, f"Path: {MODEL_PATH.name}")
except Exception as e:
    check("A1: Model file loads without error", False, str(e))
    print("Cannot continue -- model missing. Run train_model.py first.")
    sys.exit(1)

# A2. Rebuild test set from NHANES (same pipeline as training)
print(f"\n{INFO} Rebuilding test dataset from NHANES ...")
try:
    demo = pd.read_sas(DATA_DIR / "P_DEMO.xpt", encoding="latin-1")
    bp   = pd.read_sas(DATA_DIR / "P_BPXO.xpt", encoding="latin-1")
    bmx  = pd.read_sas(DATA_DIR / "P_BMX.xpt",  encoding="latin-1")
    bpq  = pd.read_sas(DATA_DIR / "P_BPQ.xpt",  encoding="latin-1")
    diq  = pd.read_sas(DATA_DIR / "P_DIQ.xpt",  encoding="latin-1")

    demo = demo[["SEQN","RIDAGEYR","RIAGENDR","RIDRETH3","DMDEDUC2","INDFMPIR"]].copy()
    demo.columns = ["SEQN","Age","Sex","Ethnicity","Education","PovertyRatio"]

    sbp_cols = ["BPXOSY1","BPXOSY2","BPXOSY3"]
    dbp_cols = ["BPXODI1","BPXODI2","BPXODI3"]
    bp = bp[["SEQN"] + sbp_cols + dbp_cols].copy()

    bmx = bmx[["SEQN","BMXBMI","BMXWAIST","BMXHT"]].copy()
    bmx.columns = ["SEQN","BMI","Waist_cm","Height_cm"]

    bpq = bpq[["SEQN","BPQ050A"]].copy() if "BPQ050A" in bpq.columns else bpq[["SEQN"]].assign(BPQ050A=np.nan)
    bpq.columns = ["SEQN","BP_Medication"]

    diq = diq[["SEQN","DIQ010"]].copy() if "DIQ010" in diq.columns else diq[["SEQN"]].assign(DIQ010=np.nan)
    diq.columns = ["SEQN","Diabetes"]

    df = (demo.merge(bp, on="SEQN", how="inner")
              .merge(bmx, on="SEQN", how="left")
              .merge(bpq, on="SEQN", how="left")
              .merge(diq, on="SEQN", how="left"))

    def avg_bp(row, cols):
        vals = [row[c] for c in cols if pd.notna(row[c]) and row[c] > 0]
        return np.mean(vals) if vals else np.nan

    df["SBP_mean"] = df.apply(avg_bp, axis=1, cols=sbp_cols)
    df["DBP_mean"] = df.apply(avg_bp, axis=1, cols=dbp_cols)
    df["elevated_bp"] = ((df["SBP_mean"] >= 130) | (df["DBP_mean"] >= 80)).astype(int)
    df = df.dropna(subset=["SBP_mean","DBP_mean"])
    df["Waist_to_Height"] = np.where(df["Height_cm"]>0, df["Waist_cm"]/df["Height_cm"], np.nan)
    df["Sex"]      = df["Sex"].map({1.0:"Male",2.0:"Female"})
    df["Ethnicity"]= df["Ethnicity"].astype("category")
    df["Education"]= df["Education"].astype("category")
    df["Diabetes"] = df["Diabetes"].map({1.0:"Yes",2.0:"No",3.0:"Borderline",7.0:np.nan,9.0:np.nan})
    df["BP_Medication"] = df["BP_Medication"].map({1.0:"Yes",2.0:"No",7.0:np.nan,9.0:np.nan})
    df = df[df["Age"] >= 18].copy()

    FEATURES = ["Age","BMI","Waist_cm","Height_cm","Waist_to_Height","PovertyRatio",
                "Sex","Ethnicity","Education","Diabetes","BP_Medication"]
    TARGET   = "elevated_bp"
    df_model = df[FEATURES + [TARGET]].dropna(subset=["Age","Sex",TARGET]).copy()

    X = df_model[FEATURES]
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    check("A2: NHANES datasets load & merge", True,
          f"{len(df_model):,} adult records, {y.mean()*100:.1f}% elevated")
except Exception as e:
    check("A2: NHANES datasets load & merge", False, str(e))
    sys.exit(1)

# A3. Both classes present in test set
class_counts = y_test.value_counts()
both_present = (0 in class_counts.index) and (1 in class_counts.index)
check("A3: Both classes in test set",
      both_present,
      f"Normal={class_counts.get(0,0):,}  Elevated={class_counts.get(1,0):,}")

# A4. Predictions
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test,  y_pred)
prec    = precision_score(y_test, y_pred, zero_division=0)
rec     = recall_score(y_test,    y_pred, zero_division=0)
f1      = f1_score(y_test,        y_pred, zero_division=0)
auc     = roc_auc_score(y_test,   y_pred_prob)
cm      = confusion_matrix(y_test, y_pred)
cr      = classification_report(y_test, y_pred, target_names=["Normal","Elevated BP"])

print(f"\n{INFO} Hold-out test set metrics:")
print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision          : {prec:.4f}")
print(f"  Recall/Sensitivity : {rec:.4f}")
print(f"  F1-Score           : {f1:.4f}")
print(f"  ROC-AUC            : {auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  [[TN={cm[0][0]:4d}  FP={cm[0][1]:4d}]")
print(f"   [FN={cm[1][0]:4d}  TP={cm[1][1]:4d}]]")
print(f"\n  Classification Report:\n{cr}")

check("A4: Accuracy > 60%",      acc  > 0.60, f"{acc:.4f}")
check("A4: Recall  > 60%",       rec  > 0.60, f"{rec:.4f}")
check("A4: ROC-AUC > 0.65",      auc  > 0.65, f"{auc:.4f}")
check("A4: F1-Score > 0.55",     f1   > 0.55, f"{f1:.4f}")
check("A4: Confusion matrix 2x2",cm.shape == (2,2), str(cm.shape))

# A5. Baseline comparison (DummyClassifier)
print(f"\n{INFO} Baseline comparison (DummyClassifier - most_frequent):")
dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
dummy.fit(X_train, y_train)
d_pred = dummy.predict(X_test)
d_acc  = accuracy_score(y_test, d_pred)
d_f1   = f1_score(y_test, d_pred, zero_division=0)
d_rec  = recall_score(y_test, d_pred, zero_division=0)
print(f"  Dummy Accuracy : {d_acc:.4f}")
print(f"  Dummy Recall   : {d_rec:.4f}")
print(f"  Dummy F1-Score : {d_f1:.4f}")
print(f"  RF Accuracy    : {acc:.4f}  (delta +{(acc-d_acc)*100:.1f}%)")
print(f"  RF Recall      : {rec:.4f}  (delta +{(rec-d_rec)*100:.1f}%)")

check("A5: RF beats Dummy on Accuracy", acc > d_acc, f"RF={acc:.4f} Dummy={d_acc:.4f}")
check("A5: RF beats Dummy on Recall",   rec > d_rec, f"RF={rec:.4f} Dummy={d_rec:.4f}")
check("A5: RF beats Dummy on F1",       f1  > d_f1,  f"RF={f1:.4f} Dummy={d_f1:.4f}")

# A6. Cross-validation stability
print(f"\n{INFO} 5-fold cross-validation (stability check) ...")
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_f1  = cross_val_score(model, X, y, cv=cv, scoring="f1",      n_jobs=-1)
cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"  F1  : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}  (folds: {[f'{v:.3f}' for v in cv_f1]})")
print(f"  AUC : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}  (folds: {[f'{v:.3f}' for v in cv_auc]})")

check("A6: CV F1 std < 0.05 (stable)",  cv_f1.std()  < 0.05, f"std={cv_f1.std():.4f}")
check("A6: CV AUC std < 0.05 (stable)", cv_auc.std() < 0.05, f"std={cv_auc.std():.4f}")
check("A6: CV AUC mean > 0.65",         cv_auc.mean()> 0.65, f"mean={cv_auc.mean():.4f}")

# A7. Data leakage check
leakage_cols = ["SBP_mean","DBP_mean","PulsePressure","MAP",
                "BPXOSY1","BPXOSY2","BPXOSY3","BPXODI1","BPXODI2","BPXODI3"]
leaked = [c for c in leakage_cols if c in X.columns]
check("A7: No raw BP columns in features (no data leakage)",
      len(leaked) == 0,
      f"Leaked columns: {leaked}" if leaked else "Clean -- no BP columns in X")


# ══════════════════════════════════════════════════════════════
# SECTION B -- DIGITAL TWIN WORKFLOW TESTS (A, B, C)
# ══════════════════════════════════════════════════════════════
print(f"\n\n[B] DIGITAL TWIN WORKFLOW TESTS\n" + SEP2)
print(f"{INFO} Initialising isolated test database ...")

# Use a temporary test DB so we don't corrupt the live DB
import database as _db_mod
_orig_db = _db_mod.DB_PATH
_db_mod.DB_PATH = DATA_DIR / "test_twin.db"
# Clean slate
if _db_mod.DB_PATH.exists():
    _db_mod.DB_PATH.unlink()
_db_mod.init_db()

# Create test patient
test_patient_id = _db_mod.create_patient({
    "name": "Test Patient Alpha", "age": 45, "sex": "Male",
    "bmi": 29.0, "waist_cm": 98.0, "height_cm": 175.0,
    "ethnicity": "3.0", "education": "4.0", "poverty_ratio": 2.8,
    "diabetes": "No", "bp_medication": "No",
})
check("B0: Test patient created in DB", test_patient_id > 0,
      f"patient_id = {test_patient_id}")

# ── Test A: Initial reading ────────────────────────────────────
print(f"\n--- Test A: Initial reading (128/82) ---")
ts_a = (datetime.now() - timedelta(days=2)).isoformat(timespec="seconds")
rid_a = _db_mod.add_reading(test_patient_id, {
    "systolic": 128, "diastolic": 82, "heart_rate": 70,
    "risk_score": 45.0, "risk_level": "Moderate",
    "notes": "Morning baseline", "timestamp": ts_a,
})
readings_after_a = _db_mod.get_readings(test_patient_id)

check("A-1: Reading stored in DB", rid_a > 0, f"reading_id = {rid_a}")
check("A-2: Exactly 1 reading retrieved", len(readings_after_a) == 1,
      f"Got {len(readings_after_a)} readings")
check("A-3: SBP stored correctly",
      abs(readings_after_a[0]["systolic"] - 128) < 0.1, "128 mmHg")
check("A-4: DBP stored correctly",
      abs(readings_after_a[0]["diastolic"] - 82) < 0.1, "82 mmHg")
check("A-5: Timestamp stored",
      readings_after_a[0]["timestamp"] == ts_a, ts_a)
check("A-6: Patient ID matches",
      readings_after_a[0]["patient_id"] == test_patient_id,
      f"pid={readings_after_a[0]['patient_id']}")

bp_label_a, is_elevated_a = te.classify_bp(128, 82)
check("A-7: ACC/AHA classification correct (Stage 1 Range)",
      "Stage 1" in bp_label_a, f"Got: {bp_label_a}")
check("A-8: Elevated flag is True for 128/82",
      is_elevated_a, "SBP 128 < 130 but DBP 82 >= 80 -> Stage 1 Range")

# ── Test B: Update reading + alerts ───────────────────────────
print(f"\n--- Test B: Second reading (145/92) + alert check ---")
ts_b = (datetime.now() - timedelta(hours=4)).isoformat(timespec="seconds")
rid_b = _db_mod.add_reading(test_patient_id, {
    "systolic": 145, "diastolic": 92, "heart_rate": 78,
    "risk_score": 61.0, "risk_level": "High",
    "notes": "Elevated reading", "timestamp": ts_b,
})
readings_after_b = _db_mod.get_readings(test_patient_id)
check("B-1: Second reading stored", rid_b > 0)
check("B-2: History now has 2 readings", len(readings_after_b) == 2,
      f"Got {len(readings_after_b)} readings")
check("B-3: Readings in chronological order",
      readings_after_b[0]["timestamp"] <= readings_after_b[1]["timestamp"])
check("B-4: Latest SBP is 145",
      abs(readings_after_b[-1]["systolic"] - 145) < 0.1)

# Alert generation — pass all readings including the latest one
new_alerts_b = te.generate_alerts(
    test_patient_id, readings_after_b,
    145, 92, 61.0, "High"
)
all_alerts_b = _db_mod.get_alerts(test_patient_id, unack_only=True)

check("B-5: Alert(s) generated for 145/92", len(all_alerts_b) >= 1,
      f"{len(all_alerts_b)} alert(s) raised")
alert_severities = [a["severity"] for a in all_alerts_b]
check("B-6: At least one warning/danger alert",
      any(s in ("warning","danger") for s in alert_severities),
      f"Severities: {alert_severities}")

bp_label_b, _ = te.classify_bp(145, 92)
check("B-7: 145/92 classified as Stage 2 Range",
      "Stage 2" in bp_label_b, f"Got: {bp_label_b}")

# ── Test C: Idempotency, rolling stats, missing values ────────
print(f"\n--- Test C: Idempotency, rolling avg, edge cases ---")

# Add 5 more readings for trend analysis
reading_vals = [
    (130, 84), (135, 86), (138, 87), (141, 89), (143, 91)
]
start = datetime.now() - timedelta(days=7)
for i, (sbp, dbp) in enumerate(reading_vals):
    ts = (start + timedelta(days=i)).isoformat(timespec="seconds")
    _db_mod.add_reading(test_patient_id, {
        "systolic": sbp, "diastolic": dbp, "heart_rate": 72,
        "risk_score": 55.0, "risk_level": "High",
        "notes": f"Day {i+1}", "timestamp": ts,
    })

all_readings = _db_mod.get_readings(test_patient_id)
check("C-1: All 7 readings stored (2 + 5)", len(all_readings) == 7,
      f"Got {len(all_readings)}")

# Trend analysis
trend = te.analyse_trend(all_readings)
check("C-2: Trend direction is computed", trend["direction"] in ("Rising","Falling","Stable"),
      f"Direction: {trend['direction']}")
check("C-3: SBP slope is a float", isinstance(trend["sbp_slope"], float),
      f"slope={trend['sbp_slope']}")
check("C-4: Average SBP computed", trend["avg_sbp"] is not None,
      f"avg SBP={trend['avg_sbp']}")

# Rising trend expected from the added data
sbp_series = [r["systolic"] for r in all_readings]
manual_slope = np.polyfit(range(len(sbp_series)), sbp_series, 1)[0]
check("C-5: Trend slope positive (rising data)", manual_slope > 0,
      f"Manual slope = {manual_slope:.3f}")
check("C-6: Trend detected as Rising", trend["direction"] == "Rising",
      f"Got: {trend['direction']}")

# Idempotency: same reading twice should yield same class
def predict_risk(p):
    row = pd.DataFrame([{
        "Age":45,"BMI":29.0,"Waist_cm":98.0,"Height_cm":175.0,
        "Waist_to_Height":98/175,"PovertyRatio":2.8,
        "Sex":"Male","Ethnicity":"3.0","Education":"4.0",
        "Diabetes":"No","BP_Medication":"No",
    }])
    return model.predict(row)[0], float(model.predict_proba(row)[0][1])

pred1, prob1 = predict_risk({})
pred2, prob2 = predict_risk({})
check("C-7: Same input -> same prediction (idempotent)",
      pred1 == pred2 and abs(prob1 - prob2) < 1e-9,
      f"pred={pred1} prob={prob1:.4f}")

# Missing value handling: partial input with NaNs
partial = pd.DataFrame([{
    "Age":45,"BMI":np.nan,"Waist_cm":98.0,"Height_cm":175.0,
    "Waist_to_Height":98/175,"PovertyRatio":np.nan,
    "Sex":"Male","Ethnicity":"3.0","Education":"4.0",
    "Diabetes":"No","BP_Medication":"No",
}])
try:
    pred_partial = model.predict(partial)[0]
    check("C-8: Model handles NaN inputs (imputer in pipeline)", True,
          f"Predicted class {pred_partial} with 2 missing values")
except Exception as e:
    check("C-8: Model handles NaN inputs", False, str(e))

# Alert acknowledgement
if all_alerts_b:
    first_alert_id = all_alerts_b[0]["id"]
    _db_mod.acknowledge_alert(first_alert_id)
    remaining = _db_mod.get_alerts(test_patient_id, unack_only=True)
    check("C-9: Alert acknowledgement removes from unack list",
          len(remaining) < len(all_alerts_b),
          f"Before={len(all_alerts_b)}  After={len(remaining)}")

# Hypertensive crisis alert rule
ts_crisis = datetime.now().isoformat(timespec="seconds")
crisis_alerts = te.generate_alerts(
    test_patient_id, all_readings, 185, 118, 90.0, "Critical"
)
check("C-10: Crisis alert fires for SBP 185 / DBP 118",
      any(a.get("severity") == "danger" for a in crisis_alerts),
      f"Crisis alerts: {crisis_alerts}")

# ── Restore original DB path ───────────────────────────────────
_db_mod.DB_PATH = _orig_db


# ══════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════
print(f"\n\n[C] SUMMARY REPORT\n" + SEP2)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
pct    = passed / total * 100 if total else 0

print(f"\n  Tests Passed : {passed} / {total}  ({pct:.0f}%)\n")
for name, ok, detail in results:
    status = PASS if ok else FAIL
    print(f"  {status}  {name}")
    if not ok and detail:
        print(f"         Detail: {detail}")

print(f"\n{SEP}")
if passed == total:
    print("  ALL TESTS PASSED -- Digital Twin validation complete!")
else:
    print(f"  {total - passed} test(s) failed -- review details above.")
print(SEP)

# Save report
lines = [
    "Hypertension Digital Twin -- Full Validation Report",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 65,
    "",
    "MODEL METRICS (Hold-out Test Set)",
    f"  Accuracy           : {acc:.4f}",
    f"  Precision          : {prec:.4f}",
    f"  Recall/Sensitivity : {rec:.4f}",
    f"  F1-Score           : {f1:.4f}",
    f"  ROC-AUC            : {auc:.4f}",
    "",
    f"  Confusion Matrix:  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}",
    "",
    "BASELINE COMPARISON",
    f"  DummyClassifier Accuracy : {d_acc:.4f}",
    f"  RF Accuracy              : {acc:.4f}  (+{(acc-d_acc)*100:.1f}% over baseline)",
    f"  RF Recall                : {rec:.4f}  (+{(rec-d_rec)*100:.1f}% over baseline)",
    "",
    "5-FOLD CROSS-VALIDATION",
    f"  F1   mean={cv_f1.mean():.4f}  std={cv_f1.std():.4f}",
    f"  AUC  mean={cv_auc.mean():.4f}  std={cv_auc.std():.4f}",
    "",
    "TEST RESULTS",
] + [
    f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else "")
    for name, ok, detail in results
] + [
    "",
    f"OVERALL: {passed}/{total} passed ({pct:.0f}%)",
    "",
    "DISCLAIMER: This report validates software behavior only.",
    "Results are not clinically validated. Not for medical use.",
]
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\n  Full report saved -> {REPORT_PATH.name}")
