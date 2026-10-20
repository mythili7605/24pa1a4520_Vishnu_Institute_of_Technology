# -*- coding: utf-8 -*-
"""
database.py – SQLite persistence layer for the Hypertension Digital Twin
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "digital_twin.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they do not exist."""
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS patients (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT    NOT NULL,
        age           REAL    NOT NULL,
        sex           TEXT    NOT NULL,
        bmi           REAL,
        waist_cm      REAL,
        height_cm     REAL,
        ethnicity     TEXT    DEFAULT '3.0',
        education     TEXT    DEFAULT '3.0',
        poverty_ratio REAL    DEFAULT 2.5,
        diabetes      TEXT    DEFAULT 'No',
        bp_medication TEXT    DEFAULT 'No',
        created_at    TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS bp_readings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id  INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        systolic    REAL    NOT NULL,
        diastolic   REAL    NOT NULL,
        heart_rate  REAL,
        risk_score  REAL,
        risk_level  TEXT,
        notes       TEXT,
        timestamp   TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id     INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        message        TEXT    NOT NULL,
        severity       TEXT    NOT NULL,
        trigger        TEXT,
        acknowledged   INTEGER DEFAULT 0,
        created_at     TEXT    DEFAULT (datetime('now'))
    );
    """)
    conn.commit()
    conn.close()


# ── Patient CRUD ──────────────────────────────────────────────────────────────

def create_patient(data: dict) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO patients
            (name, age, sex, bmi, waist_cm, height_cm,
             ethnicity, education, poverty_ratio, diabetes, bp_medication)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data["name"], data["age"], data["sex"],
        data.get("bmi"), data.get("waist_cm"), data.get("height_cm"),
        data.get("ethnicity", "3.0"), data.get("education", "3.0"),
        data.get("poverty_ratio", 2.5),
        data.get("diabetes", "No"), data.get("bp_medication", "No"),
    ))
    pid = c.lastrowid
    conn.commit()
    conn.close()
    return pid


def list_patients() -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT
            p.*,
            COUNT(r.id) AS reading_count,
            (
                SELECT r2.timestamp
                FROM bp_readings r2
                WHERE r2.patient_id = p.id
                ORDER BY r2.timestamp DESC, r2.id DESC
                LIMIT 1
            ) AS last_reading_at,
            (
                SELECT r2.systolic
                FROM bp_readings r2
                WHERE r2.patient_id = p.id
                ORDER BY r2.timestamp DESC, r2.id DESC
                LIMIT 1
            ) AS last_sbp,
            (
                SELECT r2.diastolic
                FROM bp_readings r2
                WHERE r2.patient_id = p.id
                ORDER BY r2.timestamp DESC, r2.id DESC
                LIMIT 1
            ) AS last_dbp,
            (
                SELECT COUNT(*)
                FROM alerts a
                WHERE a.patient_id = p.id
                  AND a.acknowledged = 0
            ) AS alert_count
        FROM patients p
        LEFT JOIN bp_readings r ON r.patient_id = p.id
        GROUP BY p.id
        ORDER BY p.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient(pid: int) -> dict:
    conn = get_conn()
    row = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def patient_exists() -> bool:
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    conn.close()
    return count > 0


# ── BP Readings CRUD ──────────────────────────────────────────────────────────

def add_reading(patient_id: int, data: dict) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO bp_readings
            (patient_id, systolic, diastolic, heart_rate,
             risk_score, risk_level, notes, timestamp)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        patient_id,
        data["systolic"], data["diastolic"],
        data.get("heart_rate"),
        data.get("risk_score"), data.get("risk_level"),
        data.get("notes", ""),
        data.get("timestamp", datetime.now().isoformat(timespec="seconds")),
    ))
    rid = c.lastrowid
    conn.commit()
    conn.close()
    return rid


def get_readings(patient_id: int) -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM bp_readings
        WHERE patient_id = ?
        ORDER BY timestamp ASC
    """, (patient_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Alerts CRUD ───────────────────────────────────────────────────────────────

def add_alert(patient_id: int, message: str, severity: str, trigger: str = ""):
    conn = get_conn()
    conn.execute("""
        INSERT INTO alerts (patient_id, message, severity, trigger)
        VALUES (?,?,?,?)
    """, (patient_id, message, severity, trigger))
    conn.commit()
    conn.close()


def get_alerts(patient_id: int, unack_only: bool = True) -> list:
    conn = get_conn()
    q = "SELECT * FROM alerts WHERE patient_id=?"
    if unack_only:
        q += " AND acknowledged=0"
    q += " ORDER BY created_at DESC"
    rows = conn.execute(q, (patient_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: int):
    conn = get_conn()
    conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()
