
import sqlite3
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "health_monitor.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        310000
    )

    return salt.hex(), password_hash.hex()


def verify_password(password, salt_hex, stored_hash):
    salt = bytes.fromhex(salt_hex)

    _, password_hash = hash_password(password, salt)

    return secrets.compare_digest(password_hash, stored_hash)


def init_db():
    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS health_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                heart_rate REAL,
                systolic_bp REAL,
                diastolic_bp REAL,
                spo2 REAL,
                temperature REAL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (patient_id)
                    REFERENCES patients(patient_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (patient_id)
                    REFERENCES patients(patient_id)
            )
        """)

        conn.commit()


def register_patient(full_name, email, age, gender, password):
    full_name = full_name.strip()
    email = email.strip().lower()

    if not full_name or not email:
        return False, "Name and email are required."

    if not 1 <= age <= 120:
        return False, "Please enter a valid age."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    salt, password_hash = hash_password(password)

    patient_id = "PAT-" + secrets.token_hex(4).upper()

    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO patients (
                    patient_id, full_name, email, age,
                    gender, salt, password_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id,
                full_name,
                email,
                age,
                gender,
                salt,
                password_hash,
                created_at
            ))

        return True, patient_id

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def authenticate_patient(email, password):
    email = email.strip().lower()

    with get_connection() as conn:
        patient = conn.execute("""
            SELECT * FROM patients
            WHERE email = ?
        """, (email,)).fetchone()

    if patient is None:
        return None

    if not verify_password(
        password,
        patient["salt"],
        patient["password_hash"]
    ):
        return None

    return dict(patient)


def get_patient(patient_id):
    with get_connection() as conn:
        patient = conn.execute("""
            SELECT patient_id, full_name, email, age,
                   gender, created_at
            FROM patients
            WHERE patient_id = ?
        """, (patient_id,)).fetchone()

    return dict(patient) if patient else None


def save_health_reading(
    patient_id,
    heart_rate=None,
    systolic_bp=None,
    diastolic_bp=None,
    spo2=None,
    temperature=None
):
    recorded_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO health_readings (
                patient_id, heart_rate, systolic_bp,
                diastolic_bp, spo2, temperature, recorded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            heart_rate,
            systolic_bp,
            diastolic_bp,
            spo2,
            temperature,
            recorded_at
        ))


def get_health_history(patient_id):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM health_readings
            WHERE patient_id = ?
            ORDER BY recorded_at DESC
        """, (patient_id,)).fetchall()

    return [dict(row) for row in rows]


def save_prediction(patient_id, symptoms, predicted_class):
    recorded_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions (
                patient_id, symptoms, predicted_class, recorded_at
            )
            VALUES (?, ?, ?, ?)
        """, (
            patient_id,
            symptoms,
            predicted_class,
            recorded_at
        ))


def get_prediction_history(patient_id):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE patient_id = ?
            ORDER BY recorded_at DESC
        """, (patient_id,)).fetchall()

    return [dict(row) for row in rows]
