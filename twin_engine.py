# -*- coding: utf-8 -*-
"""
twin_engine.py – Trend analysis and alert generation for the Digital Twin
"""

import numpy as np
from datetime import datetime, timedelta
from database import get_readings, add_alert, get_alerts


# ─────────────────────────────────────────────────────────────────────────────
# BP Classification helpers (ACC/AHA 2017)
# ─────────────────────────────────────────────────────────────────────────────

def classify_bp(sbp, dbp):
    """Return (BP category, elevated_bool).

    Screening categories based on ACC/AHA 2017 thresholds.
    The category is determined by whichever of the systolic or
    diastolic values falls into the higher category.

    NOTE: A single reading does not establish a hypertension diagnosis.
    These are BP screening categories, not confirmed diagnoses.
    """
    sbp = float(sbp)
    dbp = float(dbp)

    if sbp < 0 or dbp < 0:
        raise ValueError("Blood pressure values cannot be negative.")

    if sbp >= 180 or dbp >= 120:
        return "Hypertensive Crisis Range", True
    if sbp >= 140 or dbp >= 90:
        return "Stage 2 Hypertension Range", True
    if sbp >= 130 or dbp >= 80:
        return "Stage 1 Hypertension Range", True
    if sbp >= 120 and dbp < 80:
        return "Elevated BP", True
    return "Normal BP", False


# ─────────────────────────────────────────────────────────────────────────────
# Trend Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_trend(readings: list, window: int = 7) -> dict:
    """
    Run linear regression on the last `window` readings.
    Returns trend direction, estimated slope (mmHg/day), and summary stats.

    Notes:
    - The slope uses the latest `window` readings (or fewer if fewer exist).
    - The average, minimum and maximum use ALL readings.
    - If readings have identical or unparseable timestamps, the slope
      defaults to 0.0 rather than producing unstable results.
    - The slope is an *estimate*; it does not guarantee future readings.
    """
    recent = readings[-window:] if len(readings) >= window else readings

    if len(recent) < 2:
        return {
            "direction": "Insufficient data",
            "icon":      "—",
            "sbp_slope": 0.0,
            "dbp_slope": 0.0,
            "avg_sbp":   None,
            "avg_dbp":   None,
            "min_sbp":   None,
            "max_sbp":   None,
            "n":         len(readings),
            "window_n":  len(recent),
        }

    # Convert timestamps to numeric day offsets
    def to_day(ts_str):
        try:
            return datetime.fromisoformat(ts_str).timestamp() / 86400
        except Exception:
            return None

    raw_days = [to_day(r["timestamp"]) for r in recent]

    # If any timestamps failed to parse, fall back to index-based spacing
    if any(d is None for d in raw_days):
        days = np.arange(len(recent), dtype=float)
    else:
        days = np.array(raw_days, dtype=float)
        days -= days[0]  # offset so first reading = day 0

    sbp_vals = np.array([r["systolic"]  for r in recent], dtype=float)
    dbp_vals = np.array([r["diastolic"] for r in recent], dtype=float)

    # Least-squares slope — returns 0 when timestamps are identical
    def slope(x, y):
        if len(x) < 2 or np.std(x) < 1e-9:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    sbp_slope = slope(days, sbp_vals)
    dbp_slope = slope(days, dbp_vals)

    THRESHOLD = 0.5  # mmHg / day
    if sbp_slope > THRESHOLD:
        direction, icon = "Rising", "up"
    elif sbp_slope < -THRESHOLD:
        direction, icon = "Falling", "down"
    else:
        direction, icon = "Stable", "right"

    all_sbp = [r["systolic"]  for r in readings]
    all_dbp = [r["diastolic"] for r in readings]

    return {
        "direction": direction,
        "icon":      icon,
        "sbp_slope": round(sbp_slope, 3),
        "dbp_slope": round(dbp_slope, 3),
        "avg_sbp":   round(float(np.mean(all_sbp)), 1),
        "avg_dbp":   round(float(np.mean(all_dbp)), 1),
        "min_sbp":   round(float(np.min(all_sbp)),  1),
        "max_sbp":   round(float(np.max(all_sbp)),  1),
        "n":         len(readings),
        "window_n":  len(recent),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Alert Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_alerts(patient_id: int, readings: list, new_sbp: float, new_dbp: float,
                    new_risk: float, new_risk_level: str):
    """
    Evaluate all alert rules after a new reading is added.

    IMPORTANT: `readings` should include the latest saved reading.
    Only fires an alert if a similar one has not been raised in the last 24h.

    NOTE: These are screening alerts based on BP categories, not
    confirmed medical diagnoses.
    """
    now = datetime.now()
    existing = get_alerts(patient_id, unack_only=False)

    def recently_alerted(trigger_key: str, hours: int = 24) -> bool:
        cutoff = now - timedelta(hours=hours)
        for a in existing:
            if a["trigger"] == trigger_key:
                try:
                    at = datetime.fromisoformat(a["created_at"])
                    if at > cutoff:
                        return True
                except Exception:
                    pass
        return False

    alerts_fired = []

    # ── Rule 1: Hypertensive crisis range ─────────────────────────────────
    if new_sbp >= 180 or new_dbp >= 120:
        key = "crisis"
        if not recently_alerted(key, hours=6):
            msg = (f"HYPERTENSIVE CRISIS RANGE: SBP {new_sbp:.0f} / DBP {new_dbp:.0f} mmHg. "
                   "If this reading is confirmed on repeat measurement, seek urgent medical advice. "
                   "Chest pain, shortness of breath, weakness, confusion, or vision changes "
                   "require emergency assessment.")
            add_alert(patient_id, msg, "danger", key)
            alerts_fired.append({"severity": "danger", "message": msg})

    # ── Rule 2: Single very high reading ──────────────────────────────────
    elif new_sbp >= 160 or new_dbp >= 100:
        key = "high_reading"
        if not recently_alerted(key, hours=12):
            msg = (f"High BP reading: SBP {new_sbp:.0f} / DBP {new_dbp:.0f} mmHg. "
                   "Clinical evaluation recommended.")
            add_alert(patient_id, msg, "danger", key)
            alerts_fired.append({"severity": "danger", "message": msg})

    # ── Rule 3: Stage 2 hypertension range (SBP >= 140 or DBP >= 90) ─────
    elif new_sbp >= 140 or new_dbp >= 90:
        key = "stage2"
        if not recently_alerted(key, hours=12):
            msg = (f"Stage 2 Hypertension Range: SBP {new_sbp:.0f} / DBP {new_dbp:.0f} mmHg. "
                   "Medical follow-up strongly recommended.")
            add_alert(patient_id, msg, "warning", key)
            alerts_fired.append({"severity": "warning", "message": msg})

    # ── Rule 4: Stage 1 elevated range (SBP >= 130 or DBP >= 80) ─────────
    elif new_sbp >= 130 or new_dbp >= 80:
        key = "stage1"
        if not recently_alerted(key, hours=24):
            msg = (f"Elevated BP screening: SBP {new_sbp:.0f} / DBP {new_dbp:.0f} mmHg "
                   f"(ACC/AHA Stage 1 range). Lifestyle modifications advised.")
            add_alert(patient_id, msg, "info", key)
            alerts_fired.append({"severity": "info", "message": msg})

    # ── Rule 5: Persistent elevation (3+ consecutive elevated) ────────────
    # The latest reading is already included in `readings`.
    recent = readings[-3:] if len(readings) >= 3 else []
    if len(recent) == 3:
        all_elevated = all(
            r["systolic"] >= 130 or r["diastolic"] >= 80 for r in recent
        )
        if all_elevated:
            key = "persistent"
            if not recently_alerted(key, hours=48):
                msg = ("Persistent elevation: 3 or more consecutive readings above "
                       "130/80 mmHg. Schedule a medical follow-up.")
                add_alert(patient_id, msg, "warning", key)
                alerts_fired.append({"severity": "warning", "message": msg})

    # ── Rule 6: Rising SBP trend ──────────────────────────────────────────
    if len(readings) >= 5:
        trend = analyse_trend(readings, window=5)
        if trend["sbp_slope"] > 1.0:
            key = "rising_trend"
            if not recently_alerted(key, hours=48):
                msg = (f"Rising SBP trend detected: +{trend['sbp_slope']:.1f} mmHg/day "
                       f"over last {trend.get('window_n', 5)} readings. Monitor closely.")
                add_alert(patient_id, msg, "warning", key)
                alerts_fired.append({"severity": "warning", "message": msg})

    # ── Rule 7: Risk score escalation ─────────────────────────────────────
    # NOTE: The RF risk score is based on patient profile features, not the
    # latest BP measurement. Since the same profile is used for each reading,
    # this alert will only trigger when profile data changes (e.g. new BMI,
    # medication status). It should not be interpreted as a real-time BP risk.
    if len(readings) >= 2:
        prev_risk = readings[-2].get("risk_score") or 0
        if new_risk is not None and (new_risk - prev_risk) > 15:
            key = "risk_jump"
            if not recently_alerted(key, hours=24):
                msg = (f"Risk score changed from {prev_risk:.1f}% to {new_risk:.1f}%. "
                       "Review updated patient profile factors.")
                add_alert(patient_id, msg, "danger", key)
                alerts_fired.append({"severity": "danger", "message": msg})

    return alerts_fired


# ─────────────────────────────────────────────────────────────────────────────
# Demo Patient A seed data
# ─────────────────────────────────────────────────────────────────────────────

def seed_demo_readings(start_date: datetime = None):
    """
    14 days of simulated BP readings for Patient A showing a gradual
    rising trend from near-normal to clearly elevated.
    Returns list of reading dicts (without patient_id).
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=14)

    readings = [
        # (day_offset, SBP, DBP, HR, notes)
        (0,  124, 78,  68, "Morning reading – baseline"),
        (1,  127, 80,  70, "After breakfast"),
        (2,  126, 79,  69, "Evening reading"),
        (3,  129, 82,  72, "After mild exercise"),
        (4,  131, 83,  71, "Felt slightly fatigued"),
        (5,  133, 84,  73, "Morning reading"),
        (6,  130, 81,  70, "After rest"),
        (7,  135, 85,  74, "1-week mark"),
        (8,  137, 87,  75, "Morning – slightly stressed"),
        (9,  138, 86,  74, "Evening reading"),
        (10, 140, 88,  76, "After work"),
        (11, 139, 89,  75, "Morning"),
        (12, 142, 90,  77, "Headache noted"),
        (13, 145, 92,  78, "Latest reading – concerning"),
    ]

    result = []
    for day_off, sbp, dbp, hr, note in readings:
        ts = (start_date + timedelta(days=day_off)).replace(
            hour=8, minute=0, second=0
        ).isoformat(timespec="seconds")
        result.append({
            "systolic":   sbp,
            "diastolic":  dbp,
            "heart_rate": hr,
            "notes":      note,
            "timestamp":  ts,
        })
    return result
