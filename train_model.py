# -*- coding: utf-8 -*-
"""
===========================================================================
Hypertension Digital Twin - Random Forest Classifier
Happiest Health 2026 Challenge  |  PoC Baseline Model
===========================================================================
Datasets  : P_DEMO.xpt  P_BPXO.xpt  P_BMX.xpt  P_BPQ.xpt  P_DIQ.xpt
Target    : elevated_bp  (SBP >= 130 OR DBP >= 80)
Features  : Age  Sex  BMI  Waist  BP-medication  Diabetes status
            + derived: waist-to-height ratio
Output    : trained model  --> rf_hypertension_model.pkl
            metrics report --> evaluation_report.txt
            plots          --> confusion_matrix.png
                              feature_importance.png
                              roc_curve.png
===========================================================================
"""

import sys, io
# Force UTF-8 output so special chars never crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")          # headless backend - safe for all environments
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, ConfusionMatrixDisplay
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_DIR     = Path(".")
OUTPUT_DIR   = Path(".")
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("=" * 65)
print("  HYPERTENSION DIGITAL TWIN - Random Forest Baseline")
print("=" * 65)
print("\n[1/6] Loading NHANES datasets ...")

demo = pd.read_sas(DATA_DIR / "P_DEMO.xpt", encoding="latin-1")
bp   = pd.read_sas(DATA_DIR / "P_BPXO.xpt", encoding="latin-1")
bmx  = pd.read_sas(DATA_DIR / "P_BMX.xpt",  encoding="latin-1")
bpq  = pd.read_sas(DATA_DIR / "P_BPQ.xpt",  encoding="latin-1")
diq  = pd.read_sas(DATA_DIR / "P_DIQ.xpt",  encoding="latin-1")

for name, df in [("DEMO", demo), ("BPXO", bp), ("BMX", bmx),
                  ("BPQ", bpq),  ("DIQ",  diq)]:
    print(f"   {name:6s}  ->  {len(df):,} rows  x  {df.shape[1]} cols")

# ---------------------------------------------------------------------------
# 2. SELECT COLUMNS
# ---------------------------------------------------------------------------
print("\n[2/6] Selecting & merging columns ...")

demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3",
             "DMDEDUC2", "INDFMPIR"]].copy()
demo.columns = ["SEQN", "Age", "Sex", "Ethnicity", "Education", "PovertyRatio"]

sbp_cols = ["BPXOSY1", "BPXOSY2", "BPXOSY3"]
dbp_cols = ["BPXODI1", "BPXODI2", "BPXODI3"]
bp = bp[["SEQN"] + sbp_cols + dbp_cols].copy()

bmx = bmx[["SEQN", "BMXBMI", "BMXWAIST", "BMXHT"]].copy()
bmx.columns = ["SEQN", "BMI", "Waist_cm", "Height_cm"]

if "BPQ050A" in bpq.columns:
    bpq = bpq[["SEQN", "BPQ050A"]].copy()
    bpq.columns = ["SEQN", "BP_Medication"]
else:
    bpq = bpq[["SEQN"]].copy()
    bpq["BP_Medication"] = np.nan

if "DIQ010" in diq.columns:
    diq = diq[["SEQN", "DIQ010"]].copy()
    diq.columns = ["SEQN", "Diabetes"]
else:
    diq = diq[["SEQN"]].copy()
    diq["Diabetes"] = np.nan

# ---------------------------------------------------------------------------
# 3. MERGE ALL DATASETS
# ---------------------------------------------------------------------------
df = (demo
      .merge(bp,  on="SEQN", how="inner")
      .merge(bmx, on="SEQN", how="left")
      .merge(bpq, on="SEQN", how="left")
      .merge(diq, on="SEQN", how="left"))

print(f"   Merged dataset  ->  {len(df):,} rows  x  {df.shape[1]} cols")

# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING & TARGET CREATION
# ---------------------------------------------------------------------------
print("\n[3/6] Engineering features & creating target ...")

def avg_bp(row, cols):
    vals = [row[c] for c in cols if pd.notna(row[c]) and row[c] > 0]
    return np.mean(vals) if vals else np.nan

df["SBP_mean"] = df.apply(avg_bp, axis=1, cols=sbp_cols)
df["DBP_mean"] = df.apply(avg_bp, axis=1, cols=dbp_cols)

# Target: ACC/AHA 2017 Stage-1+ elevated BP  (SBP>=130 OR DBP>=80)
df["elevated_bp"] = (
    (df["SBP_mean"] >= 130) | (df["DBP_mean"] >= 80)
).astype(int)

df = df.dropna(subset=["SBP_mean", "DBP_mean"])

# Waist-to-height ratio (derived, no BP leakage)
df["Waist_to_Height"] = np.where(
    df["Height_cm"] > 0,
    df["Waist_cm"] / df["Height_cm"],
    np.nan
)

# Encode
df["Sex"]       = df["Sex"].map({1.0: "Male", 2.0: "Female"})
df["Ethnicity"] = df["Ethnicity"].astype("category")
df["Education"] = df["Education"].astype("category")
df["Diabetes"]  = df["Diabetes"].map({1.0: "Yes", 2.0: "No",
                                       3.0: "Borderline",
                                       7.0: np.nan, 9.0: np.nan})
df["BP_Medication"] = df["BP_Medication"].map({1.0: "Yes", 2.0: "No",
                                                7.0: np.nan, 9.0: np.nan})

# Adults only
df = df[df["Age"] >= 18].copy()
vc = df["elevated_bp"].value_counts()
print(f"   Adults (>=18 yrs) in merged dataset  ->  {len(df):,}")
print(f"   Class balance - elevated_bp:")
print(f"     0 (Normal)   : {vc.get(0,0):,}  ({vc.get(0,0)/len(df)*100:.1f}%)")
print(f"     1 (Elevated) : {vc.get(1,0):,}  ({vc.get(1,0)/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 5. DEFINE FEATURE SETS
# ---------------------------------------------------------------------------
# NOTE: SBP_mean / DBP_mean are excluded to prevent data leakage
#       (they were used to CREATE the target label)

NUMERIC_FEATURES     = ["Age", "BMI", "Waist_cm", "Height_cm",
                         "Waist_to_Height", "PovertyRatio"]
CATEGORICAL_FEATURES = ["Sex", "Ethnicity", "Education",
                         "Diabetes", "BP_Medication"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET               = "elevated_bp"

df_model = df[ALL_FEATURES + [TARGET]].dropna(
    subset=["Age", "Sex", TARGET]
).copy()
print(f"   Final model dataset                ->  {len(df_model):,} rows")

X = df_model[ALL_FEATURES]
y = df_model[TARGET]

# ---------------------------------------------------------------------------
# 6. BUILD sklearn PIPELINE
# ---------------------------------------------------------------------------
print("\n[4/6] Building Random Forest pipeline ...")

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,      NUMERIC_FEATURES),
    ("cat", categorical_transformer,  CATEGORICAL_FEATURES)
])

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators     = 300,
        max_depth        = None,
        min_samples_leaf = 4,
        max_features     = "sqrt",
        class_weight     = "balanced",   # handles class imbalance
        random_state     = RANDOM_STATE,
        n_jobs           = -1
    ))
])

# ---------------------------------------------------------------------------
# 7. TRAIN / EVALUATE
# ---------------------------------------------------------------------------
print("\n[5/6] Training & evaluating ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y
)
print(f"   Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")

rf_pipeline.fit(X_train, y_train)

y_pred      = rf_pipeline.predict(X_test)
y_pred_prob = rf_pipeline.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test,  y_pred)
prec    = precision_score(y_test, y_pred, zero_division=0)
rec     = recall_score(y_test,    y_pred, zero_division=0)
f1      = f1_score(y_test,        y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test,   y_pred_prob)
cm      = confusion_matrix(y_test, y_pred)
cr      = classification_report(y_test, y_pred,
                                  target_names=["Normal", "Elevated BP"])

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_f1   = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="f1",      n_jobs=-1)
cv_auc  = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

# ---------------------------------------------------------------------------
# 8. PRINT RESULTS
# ---------------------------------------------------------------------------
sep = "-" * 65
print(f"\n{sep}")
print("  MODEL EVALUATION RESULTS  (Hold-out test set)")
print(sep)
print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision          : {prec:.4f}")
print(f"  Recall/Sensitivity : {rec:.4f}")
print(f"  F1-Score           : {f1:.4f}")
print(f"  ROC-AUC            : {roc_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  {cm}")
print(f"\n  Classification Report:\n{cr}")
print(f"\n  5-Fold Cross-Validation:")
print(f"  F1    -> mean {cv_f1.mean():.4f}  +/-  {cv_f1.std():.4f}")
print(f"  AUC   -> mean {cv_auc.mean():.4f}  +/-  {cv_auc.std():.4f}")
print(sep)

# Feature importances
rf_clf    = rf_pipeline.named_steps["classifier"]
ohe       = (rf_pipeline.named_steps["preprocessor"]
             .named_transformers_["cat"]
             .named_steps["onehot"])
cat_names     = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
feature_names = NUMERIC_FEATURES + cat_names
importances   = rf_clf.feature_importances_

top_n = min(15, len(feature_names))
idx   = np.argsort(importances)[::-1][:top_n]
print(f"\n  Top-{top_n} Feature Importances:")
for rank, i in enumerate(idx, 1):
    print(f"  {rank:2d}. {feature_names[i]:<35s}  {importances[i]:.4f}")
print(sep)

# ---------------------------------------------------------------------------
# 9. SAVE OUTPUTS
# ---------------------------------------------------------------------------
print("\n[6/6] Saving model & plots ...")

# 9a. Persist model
model_path = OUTPUT_DIR / "rf_hypertension_model.pkl"
joblib.dump(rf_pipeline, model_path)
print(f"   Model saved  -> {model_path}")

# 9b. Metrics text report
report_lines = [
    "Hypertension Digital Twin - RF Baseline Evaluation",
    "=" * 60,
    f"Accuracy           : {acc:.4f}",
    f"Precision          : {prec:.4f}",
    f"Recall             : {rec:.4f}",
    f"F1-Score           : {f1:.4f}",
    f"ROC-AUC            : {roc_auc:.4f}",
    "",
    "Confusion Matrix:",
    str(cm),
    "",
    "Classification Report:",
    cr,
    "",
    "5-Fold Cross-Validation:",
    f"  F1    mean={cv_f1.mean():.4f}  std={cv_f1.std():.4f}",
    f"  AUC   mean={cv_auc.mean():.4f}  std={cv_auc.std():.4f}",
]
report_path = OUTPUT_DIR / "evaluation_report.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"   Report saved -> {report_path}")

# 9c. Confusion Matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Normal", "Elevated BP"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix - Random Forest\n(Hypertension Digital Twin)", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print("   Plot saved   -> confusion_matrix.png")

# 9d. Feature Importance plot
top_idx = np.argsort(importances)[::-1][:top_n]
fig, ax = plt.subplots(figsize=(9, 5))
colors  = sns.color_palette("viridis", top_n)
bars = ax.barh(range(top_n), importances[top_idx][::-1], color=colors[::-1])
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_names[i] for i in top_idx][::-1], fontsize=9)
ax.set_xlabel("Importance (Mean Decrease in Impurity)", fontsize=10)
ax.set_title(f"Top-{top_n} Feature Importances - Random Forest", fontsize=12)
for bar, val in zip(bars, importances[top_idx][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
plt.close()
print("   Plot saved   -> feature_importance.png")

# 9e. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#4C72B0", lw=2,
        label=f"Random Forest (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
ax.fill_between(fpr, tpr, alpha=0.15, color="#4C72B0")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
ax.set_title("ROC Curve - Hypertension Prediction", fontsize=12)
ax.legend(loc="lower right")
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
plt.close()
print("   Plot saved   -> roc_curve.png")

print("\n" + "=" * 65)
print("  DONE! Training complete. All files saved to:", OUTPUT_DIR.resolve())
print("=" * 65)
