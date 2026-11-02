
import json
import os
import re
import html
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

import joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import database

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "disease_model_augmented.joblib"

st.set_page_config(
    page_title="AI Health Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

database.init_db()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def run_model_benchmark():
    """Evaluate standard classifiers using stratified CV on Training.csv only."""
    training_path = BASE_DIR / "dataset" / "Training.csv"
    if not training_path.exists():
        raise FileNotFoundError(
            "dataset/Training.csv was not found. Add the original training dataset."
        )

    data = pd.read_csv(training_path)
    target_candidates = [
        column for column in data.columns
        if str(column).strip().lower() in {"prognosis", "disease", "target", "label"}
    ]
    if not target_candidates:
        raise ValueError("Could not identify the disease-label column in Training.csv.")

    target_column = target_candidates[0]
    X = data.drop(columns=[target_column]).fillna(0)
    y = data[target_column].astype(str).str.strip()

    if X.empty or y.nunique() < 2:
        raise ValueError("Training.csv does not contain enough features or classes.")

    min_class_count = int(y.value_counts().min())
    folds = min(5, min_class_count)
    if folds < 2:
        raise ValueError("At least two examples per disease class are needed for CV.")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced",
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", class_weight="balanced")
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    results = []
    for name, estimator in models.items():
        scores = cross_validate(
            estimator, X, y, cv=cv, scoring=scoring,
            n_jobs=1, return_train_score=False, error_score="raise"
        )
        results.append({
            "Model": name,
            "Accuracy": float(scores["test_accuracy"].mean()),
            "Precision (macro)": float(scores["test_precision_macro"].mean()),
            "Recall (macro)": float(scores["test_recall_macro"].mean()),
            "F1-score (macro)": float(scores["test_f1_macro"].mean()),
            "F1 std.": float(scores["test_f1_macro"].std()),
        })

    return pd.DataFrame(results).sort_values(
        "F1-score (macro)", ascending=False
    ).reset_index(drop=True), len(data), X.shape[1], y.nunique(), folds


if "patient" not in st.session_state:
    st.session_state.patient = None


# ---------------- STYLING ----------------

st.markdown("""
<style>
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    padding: 1.5rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #102b46, #176b70);
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 { color: white; }
.hero p { color: #e1f5f4; }
.patient-card {
    border: 1px solid rgba(128,128,128,0.35);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🏥 AI Health Monitor</h1>
    <p>Smart Patient Health Monitoring & Early Disease Prediction</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Academic prototype only. This system is not a medical device "
    "and must not be used to diagnose or treat diseases."
)


# ---------------- AUTHENTICATION ----------------

def authentication_page():

    st.title("Welcome to AI Health Monitor")
    st.write("Register as a patient or log in to access your dashboard.")

    login_tab, register_tab = st.tabs([
        "Patient Login",
        "New Patient Registration"
    ])

    with login_tab:

        st.subheader("Patient Login")

        with st.form("login_form"):

            email = st.text_input("Email address")
            password = st.text_input("Password", type="password")

            login_button = st.form_submit_button(
                "Login",
                type="primary",
                use_container_width=True
            )

        if login_button:

            if not email.strip() or not password:
                st.error("Please enter your email and password.")

            else:
                patient = database.authenticate_patient(email, password)

                if patient:
                    st.session_state.patient = database.get_patient(
                        patient["patient_id"]
                    )
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with register_tab:

        st.subheader("Create Patient Account")

        with st.form("registration_form"):

            full_name = st.text_input("Full Name")
            email = st.text_input("Email Address")

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input(
                    "Age", min_value=1, max_value=120, value=18
                )

            with col2:
                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female", "Other", "Prefer not to say"]
                )

            new_password = st.text_input(
                "Create Password", type="password"
            )

            confirm_password = st.text_input(
                "Confirm Password", type="password"
            )

            register_button = st.form_submit_button(
                "Create Account",
                type="primary",
                use_container_width=True
            )

        if register_button:

            if not full_name.strip() or not email.strip():
                st.error("Please fill in your name and email.")

            elif "@" not in email or "." not in email.split("@")[-1]:
                st.error("Please enter a valid email address.")

            elif new_password != confirm_password:
                st.error("Passwords do not match.")

            elif len(new_password) < 8:
                st.error("Password must contain at least 8 characters.")

            else:

                success, result = database.register_patient(
                    full_name=full_name,
                    email=email,
                    age=int(age),
                    gender=gender,
                    password=new_password
                )

                if success:
                    st.success("Registration successful!")
                    st.info(f"Your Patient ID is: {result}")
                    st.write("Please log in to access your account.")
                else:
                    st.error(result)


# ---------------- HEALTH READING WARNINGS ----------------

def get_health_warnings(heart_rate, systolic, diastolic, spo2, temperature):

    warnings = []
    notes = []

    # General adult resting reference ranges.
    # These are not personalized medical limits.

    if heart_rate < 60 or heart_rate > 100:
        warnings.append(
            "Heart rate is outside the typical adult resting range "
            "of 60–100 BPM."
        )

    if systolic >= 130 or diastolic >= 80:
        warnings.append(
            "Blood pressure is above the general reference threshold "
            "of 130/80 mmHg. Repeated measurements and clinical "
            "assessment are needed."
        )

    elif systolic < 90 or diastolic < 60:
        warnings.append(
            "Blood pressure is below the general reference threshold "
            "of 90/60 mmHg."
        )

    if spo2 < 95:
        warnings.append(
            "SpO2 is below the typical reference range of 95–100%. "
            "Recheck the measurement and seek medical advice."
        )

    if temperature < 36.1 or temperature > 37.2:
        warnings.append(
            "Temperature is outside the typical reference range. "
            "Interpret it according to measurement method and symptoms."
        )

    if not warnings:
        notes.append(
            "All entered values are within the selected general "
            "reference ranges. This does not guarantee good health."
        )

    return warnings, notes



def build_patient_health_pdf(patient, health_history, prediction_history):
    """Create a compact, readable educational health summary PDF in memory."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=12 * mm,
        title="Patient Health Summary"
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportSmall",
        parent=styles["BodyText"],
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#555555"),
        spaceAfter=1
    ))
    styles.add(ParagraphStyle(
        name="ReportCell",
        parent=styles["BodyText"],
        fontSize=6.5,
        leading=8,
        wordWrap="CJK"
    ))
    styles.add(ParagraphStyle(
        name="ReportHeader",
        parent=styles["BodyText"],
        fontSize=6.5,
        leading=8,
        textColor=colors.white
    ))
    story = []

    def safe(value):
        if value is None or str(value).strip() == "":
            return "Not recorded"
        return html.escape(str(value))

    def short_datetime(value):
        if value is None or str(value).strip() == "":
            return "Not recorded"
        raw = str(value).strip()
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed = parsed.astimezone(timezone.utc)
            return parsed.strftime("%d %b %Y<br/>%I:%M %p")
        except (ValueError, TypeError):
            return safe(raw[:25])

    def cell(value):
        return Paragraph(safe(value), styles["ReportCell"])

    def style_table(table, header=True, font_size=6.5):
        commands = [
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D0D9")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ]
        if header:
            commands.extend([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174B63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ])
        table.setStyle(TableStyle(commands))

    story.append(Paragraph("AI Health Monitor", styles["Title"]))
    story.append(Paragraph("Patient Health Summary Report", styles["Heading2"]))
    story.append(Paragraph(
        "Academic prototype only. This report is not a medical diagnosis, "
        "medical device output, or treatment recommendation.",
        styles["ReportSmall"]
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Patient Profile", styles["Heading2"]))
    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %I:%M %p UTC")
    profile_rows = [
        [cell("Patient name"), cell(patient.get("full_name"))],
        [cell("Patient ID"), cell(patient.get("patient_id"))],
        [cell("Age"), cell(patient.get("age"))],
        [cell("Gender"), cell(patient.get("gender"))],
        [cell("Report generated (UTC)"), cell(generated)]
    ]
    profile_table = Table(profile_rows, colWidths=[42 * mm, 130 * mm])
    style_table(profile_table, header=False)
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F5"))
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 5))

    story.append(Paragraph("Health Readings", styles["Heading2"]))
    if health_history:
        health_rows = [[
            Paragraph("Recorded at (UTC)", styles["ReportHeader"]),
            Paragraph("Heart rate<br/>(BPM)", styles["ReportHeader"]),
            Paragraph("Systolic<br/>BP", styles["ReportHeader"]),
            Paragraph("Diastolic<br/>BP", styles["ReportHeader"]),
            Paragraph("SpO2<br/>(%)", styles["ReportHeader"]),
            Paragraph("Temperature<br/>(°C)", styles["ReportHeader"])
        ]]
        for item in health_history[:30]:
            health_rows.append([
                Paragraph(short_datetime(item.get("recorded_at")), styles["ReportCell"]),
                cell(item.get("heart_rate")),
                cell(item.get("systolic_bp")),
                cell(item.get("diastolic_bp")),
                cell(item.get("spo2")),
                cell(item.get("temperature"))
            ])
        health_table = Table(
            health_rows,
            repeatRows=1,
            colWidths=[43 * mm, 24 * mm, 24 * mm, 24 * mm, 22 * mm, 35 * mm]
        )
        style_table(health_table)
        story.append(health_table)
        story.append(Spacer(1, 2))
        story.append(Paragraph(
            "Readings are manually entered by the patient and are not live sensor measurements.",
            styles["ReportSmall"]
        ))
    else:
        story.append(Paragraph("No health readings have been saved.", styles["BodyText"]))

    story.append(Spacer(1, 5))
    story.append(Paragraph("Prediction History", styles["Heading2"]))
    if prediction_history:
        prediction_rows = [[
            Paragraph("Recorded at (UTC)", styles["ReportHeader"]),
            Paragraph("Confirmed symptoms", styles["ReportHeader"]),
            Paragraph("Model output (technical only)", styles["ReportHeader"])
        ]]
        for item in prediction_history[:20]:
            prediction_rows.append([
                Paragraph(short_datetime(item.get("recorded_at")), styles["ReportCell"]),
                Paragraph(safe(item.get("symptoms")), styles["ReportCell"]),
                Paragraph(safe(item.get("predicted_class")), styles["ReportCell"])
            ])
        prediction_table = Table(
            prediction_rows,
            repeatRows=1,
            colWidths=[38 * mm, 78 * mm, 56 * mm]
        )
        style_table(prediction_table)
        story.append(prediction_table)
    else:
        story.append(Paragraph("No prediction history has been saved.", styles["BodyText"]))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Important:</b> Disease model outputs and AI-generated content are "
        "experimental educational information only. They are not clinically "
        "validated and must not be used to diagnose, treat, or rule out any "
        "condition. Consult a qualified healthcare professional for health "
        "concerns. For emergencies, contact local emergency services.",
        styles["ReportSmall"]
    ))
    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- PATIENT DASHBOARD ----------------

def patient_dashboard():

    patient = st.session_state.patient

    st.sidebar.title("Patient Dashboard")
    st.sidebar.success("Logged in")
    st.sidebar.write(f"**{patient['full_name']}**")
    st.sidebar.caption(patient["patient_id"])

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.patient = None
        st.rerun()

    st.title(f"Welcome, {patient['full_name']} 👋")

    st.caption(
        f"Patient ID: {patient['patient_id']} | "
        f"Age: {patient['age']} | Gender: {patient['gender']}"
    )

    profile_tab, monitoring_tab, prediction_tab, history_tab, assistant_tab, benchmark_tab, report_tab = st.tabs([
        "Patient Profile",
        "Health Monitoring",
        "Disease Prediction",
        "Prediction History",
        "AI Health Assistant",
        "ML Benchmarking",
        "PDF Health Report"
    ])

    # ---------------- PROFILE ----------------

    with profile_tab:

        st.subheader("Your Patient Profile")

        col1, col2, col3 = st.columns(3)

        col1.metric("Patient ID", patient["patient_id"])
        col2.metric("Age", patient["age"])
        col3.metric("Gender", patient["gender"])

        safe_name = html.escape(str(patient["full_name"]))
        safe_email = html.escape(str(patient["email"]))
        safe_id = html.escape(str(patient["patient_id"]))

        st.markdown(
            f"""
            <div class="patient-card">
                <h4>Personal Information</h4>
                <p><b>Name:</b> {safe_name}</p>
                <p><b>Email:</b> {safe_email}</p>
                <p><b>Patient ID:</b> {safe_id}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info("Your patient account is connected to the local database.")

    # ---------------- HEALTH MONITORING ----------------

    with monitoring_tab:

        st.subheader("Smart Health Monitoring")

        st.caption(
            "Enter your health readings below. These are manually "
            "entered measurements, not live sensor readings."
        )

        with st.form("health_reading_form"):

            st.markdown("#### Heart Rate")

            heart_rate = st.number_input(
                "Heart Rate (BPM)",
                min_value=40,
                max_value=180,
                value=75
            )

            st.markdown("#### Blood Pressure")

            col1, col2 = st.columns(2)

            with col1:
                systolic = st.number_input(
                    "Systolic Pressure (mmHg)",
                    min_value=60,
                    max_value=250,
                    value=120
                )

            with col2:
                diastolic = st.number_input(
                    "Diastolic Pressure (mmHg)",
                    min_value=30,
                    max_value=150,
                    value=80
                )

            st.markdown("#### Oxygen Saturation")

            spo2 = st.number_input(
                "SpO2 (%)",
                min_value=50,
                max_value=100,
                value=98
            )

            st.markdown("#### Body Temperature")

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=30.0,
                max_value=45.0,
                value=36.8,
                step=0.1,
                format="%.1f"
            )

            save_reading = st.form_submit_button(
                "Save Health Reading",
                type="primary",
                use_container_width=True
            )

        if save_reading:

            if systolic <= diastolic:
                st.error(
                    "Systolic pressure should be higher than "
                    "diastolic pressure. Please verify your readings."
                )

            else:

                database.save_health_reading(
                    patient_id=patient["patient_id"],
                    heart_rate=float(heart_rate),
                    systolic_bp=float(systolic),
                    diastolic_bp=float(diastolic),
                    spo2=float(spo2),
                    temperature=float(temperature)
                )

                st.session_state["health_reading_saved"] = True
                st.rerun()

        if st.session_state.get("health_reading_saved", False):

            st.success("Health reading saved successfully!")
            st.session_state["health_reading_saved"] = False

        st.divider()

        st.subheader("Latest Health Readings")

        health_history = database.get_health_history(
            patient["patient_id"]
        )

        if health_history:

            latest = health_history[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Heart Rate",
                f"{latest['heart_rate']:.0f} BPM"
            )

            col2.metric(
                "Blood Pressure",
                f"{latest['systolic_bp']:.0f}/"
                f"{latest['diastolic_bp']:.0f} mmHg"
            )

            col3.metric(
                "SpO2",
                f"{latest['spo2']:.0f}%"
            )

            col4, col5 = st.columns(2)

            col4.metric(
                "Temperature",
                f"{latest['temperature']:.1f} °C"
            )

            col5.metric(
                "Recorded",
                pd.to_datetime(
                    latest["recorded_at"], utc=True
                ).strftime("%d %b %Y, %I:%M %p UTC")
            )

            warnings, notes = get_health_warnings(
                latest["heart_rate"],
                latest["systolic_bp"],
                latest["diastolic_bp"],
                latest["spo2"],
                latest["temperature"]
            )

            st.markdown("#### Health Reading Review")

            if warnings:
                for warning in warnings:
                    st.warning(warning)
            else:
                for note in notes:
                    st.success(note)

            st.caption(
                "Reference ranges are general adult screening references, "
                "not personalized medical advice. A single reading cannot "
                "establish a medical condition."
            )

            # ---------------- HEALTH CHARTS ----------------

            st.divider()
            st.subheader("Health Trends")

            chart_df = pd.DataFrame(health_history)

            chart_df["recorded_at"] = pd.to_datetime(
                chart_df["recorded_at"], utc=True
            )

            chart_df = chart_df.sort_values("recorded_at")

            chart_df = chart_df.set_index("recorded_at")

            chart_tab1, chart_tab2, chart_tab3 = st.tabs([
                "Heart Rate",
                "Blood Pressure",
                "SpO2 & Temperature"
            ])

            with chart_tab1:
                st.line_chart(
                    chart_df[["heart_rate"]],
                    y_label="Beats per minute"
                )

            with chart_tab2:
                st.line_chart(
                    chart_df[["systolic_bp", "diastolic_bp"]],
                    y_label="mmHg"
                )

            with chart_tab3:

                st.line_chart(
                    chart_df[["spo2"]],
                    y_label="Oxygen saturation (%)"
                )

                st.line_chart(
                    chart_df[["temperature"]],
                    y_label="Temperature (°C)"
                )

            # ---------------- PERSONALIZED HEALTH ANALYTICS ----------------

            st.divider()
            st.subheader("Personalized Health Analytics")
            st.caption(
                "Summaries are calculated from your manually entered readings. "
                "They describe changes in your records and are not medical assessments."
            )

            period_label = st.selectbox(
                "Chart time period",
                ["Last 7 days", "Last 30 days", "Last 90 days", "All available readings"],
                index=1,
                key="health_trend_period"
            )

            latest_timestamp = chart_df.index.max()
            if period_label == "Last 7 days":
                period_start = latest_timestamp - pd.Timedelta(days=7)
            elif period_label == "Last 30 days":
                period_start = latest_timestamp - pd.Timedelta(days=30)
            elif period_label == "Last 90 days":
                period_start = latest_timestamp - pd.Timedelta(days=90)
            else:
                period_start = chart_df.index.min()

            period_df = chart_df.loc[chart_df.index >= period_start].copy()

            if period_df.empty:
                st.info("No readings are available in this selected period.")
            else:
                st.markdown("#### Selected-period averages")
                average_columns = st.columns(5)
                average_metrics = [
                    ("Heart rate", "heart_rate", "BPM", 0),
                    ("Systolic BP", "systolic_bp", "mmHg", 0),
                    ("Diastolic BP", "diastolic_bp", "mmHg", 0),
                    ("SpO2", "spo2", "%", 1),
                    ("Temperature", "temperature", "°C", 1),
                ]
                for column, (label, field, unit, decimals) in zip(
                    average_columns, average_metrics
                ):
                    average_value = period_df[field].mean()
                    column.metric(
                        label,
                        f"{average_value:.{decimals}f} {unit}"
                    )

                st.caption(
                    f"Based on {len(period_df)} reading(s), from "
                    f"{period_df.index.min().strftime('%d %b %Y')} to "
                    f"{period_df.index.max().strftime('%d %b %Y')}."
                )

                st.markdown("#### Recent 7-day comparison")
                latest_day = chart_df.index.max().normalize()
                recent_start = latest_day - pd.Timedelta(days=6)
                previous_start = latest_day - pd.Timedelta(days=13)
                recent_week = chart_df.loc[
                    (chart_df.index >= recent_start)
                    & (chart_df.index < latest_day + pd.Timedelta(days=1))
                ]
                previous_week = chart_df.loc[
                    (chart_df.index >= previous_start)
                    & (chart_df.index < recent_start)
                ]

                if previous_week.empty:
                    st.info(
                        "Add readings covering an earlier 7-day period to see "
                        "a week-to-week comparison."
                    )
                else:
                    comparison_columns = st.columns(5)
                    for column, (label, field, unit, decimals) in zip(
                        comparison_columns, average_metrics
                    ):
                        current_average = (
                            recent_week[field].mean() if not recent_week.empty else None
                        )
                        previous_average = previous_week[field].mean()
                        if current_average is None or pd.isna(current_average):
                            column.metric(label, "No recent data")
                        else:
                            difference = current_average - previous_average
                            column.metric(
                                label,
                                f"{current_average:.{decimals}f} {unit}",
                                delta=f"{difference:+.{decimals}f} {unit} vs prior 7 days",
                                delta_color="off"
                            )

                    st.caption(
                        "The comparison is descriptive only. An increase or decrease "
                        "is not automatically good or bad; interpret readings with a "
                        "qualified healthcare professional."
                    )

                st.markdown("#### Trends for selected period")
                selected_chart = period_df[
                    ["heart_rate", "systolic_bp", "diastolic_bp", "spo2", "temperature"]
                ]
                trend_tab1, trend_tab2, trend_tab3 = st.tabs([
                    "Heart Rate & BP", "SpO2", "Temperature"
                ])
                with trend_tab1:
                    st.line_chart(
                        selected_chart[["heart_rate"]],
                        y_label="Beats per minute"
                    )
                    st.line_chart(
                        selected_chart[["systolic_bp", "diastolic_bp"]],
                        y_label="mmHg"
                    )
                with trend_tab2:
                    st.line_chart(
                        selected_chart[["spo2"]],
                        y_label="Oxygen saturation (%)"
                    )
                with trend_tab3:
                    st.line_chart(
                        selected_chart[["temperature"]],
                        y_label="Temperature (°C)"
                    )

            st.markdown("#### All Saved Health Readings")

            display_df = pd.DataFrame(health_history)

            display_df["recorded_at"] = pd.to_datetime(
                display_df["recorded_at"], utc=True
            ).dt.strftime("%d %b %Y, %I:%M %p UTC")

            display_df = display_df.rename(columns={
                "heart_rate": "Heart Rate (BPM)",
                "systolic_bp": "Systolic (mmHg)",
                "diastolic_bp": "Diastolic (mmHg)",
                "spo2": "SpO2 (%)",
                "temperature": "Temperature (°C)",
                "recorded_at": "Recorded At"
            })

            st.dataframe(
                display_df[
                    [
                        "Recorded At",
                        "Heart Rate (BPM)",
                        "Systolic (mmHg)",
                        "Diastolic (mmHg)",
                        "SpO2 (%)",
                        "Temperature (°C)"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No health readings found. Enter your first reading "
                "using the form above."
            )

    # ---------------- DISEASE PREDICTION ----------------

    with prediction_tab:

        st.subheader("AI Disease Prediction")

        if not MODEL_PATH.exists():
            st.error("Disease prediction model not found.")
            st.stop()

        model = load_model()
        features = list(model.feature_names_in_)

        st.caption(
            "Describe your symptoms in English or Hinglish, "
            "or select them manually."
        )

        description = st.text_area(
            "Describe your symptoms",
            placeholder="Example: Mujhe khujli ho rahi hai aur skin par daane hain",
            height=100
        )

        with st.expander("Select Symptoms Manually"):

            search = st.text_input(
                "Search symptoms",
                key="symptom_search"
            )

            filtered_features = [
                feature for feature in features
                if search.lower() in feature.lower()
            ]

            selected = st.multiselect(
                "Choose symptoms",
                options=filtered_features,
                format_func=lambda x: x.replace("_", " ").title()
            )

        if st.button("Analyze Symptoms", type="primary", key="analyze_symptoms"):
            selected_symptoms = set(selected)
            extracted_symptoms = set()
            extraction_uncertain = False

            if description.strip():
                if not os.getenv("OPENROUTER_API_KEY"):
                    st.error(
                        "OpenRouter API key is missing. Set OPENROUTER_API_KEY "
                        "in the same CMD window used to launch Streamlit."
                    )
                    st.stop()

                prompt = f"""
You are a strict symptom phrase matcher for a college demonstration.
The patient may describe symptoms in English, Hindi, or Hinglish written
in Latin script. Understand common expressions such as:
- bukhaar / bukhar / fever -> fever
- zukham / sardi / cold -> chills, runny_nose, congestion or other exact
  labels ONLY if the description clearly supports them
- ulti / vomiting -> vomiting
- khujli -> itching
- daane / rashes -> skin_rash
- thakan / kamzori -> fatigue / weakness
- sir dard -> headache
- pet dard -> stomach_pain

Important:
1. Extract every symptom explicitly stated, not just one.
2. Do not infer symptoms that are not stated.
3. Match only exact labels from the allowed labels below.
4. If a phrase does not clearly match an exact label, omit it and set uncertain=true.
5. Return valid JSON only, with no markdown or explanation.

Allowed exact symptom labels:
{", ".join(features)}

Patient description:
{description}

Required JSON format:
{{"symptoms": ["exact_label_1", "exact_label_2"], "uncertain": false}}
"""

                try:
                    client = OpenAI(
                        api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url="https://openrouter.ai/api/v1"
                    )
                    response = client.chat.completions.create(
                        model="openrouter/free",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You extract all explicitly stated symptoms "
                                    "from English, Hindi and Hinglish. Return JSON only."
                                )
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
                    )

                    raw = (response.choices[0].message.content or "").strip()
                    raw = re.sub(
                        r"^```(?:json)?\\s*|\\s*```$",
                        "",
                        raw,
                        flags=re.IGNORECASE
                    )
                    result = json.loads(raw)

                    extracted_symptoms = {
                        symptom.strip()
                        for symptom in result.get("symptoms", [])
                        if isinstance(symptom, str)
                        and symptom.strip() in features
                    }
                    extraction_uncertain = bool(result.get("uncertain", False))

                except Exception as exc:
                    st.error(
                        "Could not extract symptoms using AI. You can still "
                        "continue with manual symptom selection. Details: " + str(exc)
                    )
                    extracted_symptoms = set()

            combined = sorted(selected_symptoms | extracted_symptoms)
            st.session_state["pending_symptoms"] = combined
            st.session_state["pending_symptom_description"] = description.strip()
            st.session_state["pending_symptom_uncertain"] = extraction_uncertain

        if "pending_symptoms" in st.session_state:
            pending = st.session_state["pending_symptoms"]

            st.divider()
            st.subheader("Confirm Extracted Symptoms")
            st.caption(
                "Review the symptoms below. Untick anything that does not apply "
                "and add any missing symptoms before prediction."
            )

            if st.session_state.get("pending_symptom_uncertain", False):
                st.warning(
                    "The AI marked part of the description as uncertain. "
                    "Please carefully review the selections."
                )

            if not pending:
                st.warning(
                    "No symptoms were matched. Select symptoms manually above "
                    "and click Analyze Symptoms again."
                )
            else:
                confirmed_symptoms = st.multiselect(
                    "Symptoms to use for prediction",
                    options=features,
                    default=pending,
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="confirmed_symptoms"
                )

                confirm_col, cancel_col = st.columns(2)

                with confirm_col:
                    predict_now = st.button(
                        "Confirm Symptoms & Predict",
                        type="primary",
                        key="confirm_symptoms_predict",
                        use_container_width=True
                    )

                with cancel_col:
                    cancel_prediction = st.button(
                        "Cancel",
                        key="cancel_pending_prediction",
                        use_container_width=True
                    )

                if cancel_prediction:
                    for key in [
                        "pending_symptoms",
                        "pending_symptom_description",
                        "pending_symptom_uncertain",
                        "confirmed_symptoms"
                    ]:
                        st.session_state.pop(key, None)
                    st.rerun()

                if predict_now:
                    all_symptoms = set(confirmed_symptoms)

                    if not all_symptoms:
                        st.error("Please confirm at least one symptom.")
                    else:
                        row = {feature: 0 for feature in features}
                        for symptom in all_symptoms:
                            row[symptom] = 1

                        input_df = pd.DataFrame([row], columns=features)
                        model = load_model()
                        prediction = str(model.predict(input_df)[0]).strip()

                        # Technical uncertainty checks. These are safeguards for
                        # an academic demo, not clinically validated thresholds.
                        top_score = None
                        score_margin = None
                        probabilities = None
                        top_predictions = []
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(input_df)[0]
                            ranked_pairs = sorted(
                                [
                                    (str(label).strip(), float(score))
                                    for label, score in zip(model.classes_, probabilities)
                                ],
                                key=lambda item: item[1],
                                reverse=True
                            )
                            top_predictions = ranked_pairs[:3]
                            ranked_scores = [score for _, score in ranked_pairs]
                            if ranked_scores:
                                top_score = ranked_scores[0]
                            if len(ranked_scores) > 1:
                                score_margin = ranked_scores[0] - ranked_scores[1]

                        # Check whether the exact confirmed symptom combination
                        # occurs in the original training data for the predicted
                        # class. This is dataset support, not medical evidence.
                        training_path = BASE_DIR / "dataset" / "Training.csv"
                        matching_rows = 0
                        predicted_class_matches = 0
                        training_check_available = False

                        if training_path.exists():
                            try:
                                training_df = pd.read_csv(training_path)
                                if "prognosis" in training_df.columns and all(
                                    symptom in training_df.columns
                                    for symptom in all_symptoms
                                ):
                                    training_check_available = True
                                    symptom_mask = pd.Series(
                                        True, index=training_df.index
                                    )
                                    for symptom in all_symptoms:
                                        symptom_mask &= (
                                            pd.to_numeric(
                                                training_df[symptom],
                                                errors="coerce"
                                            ).fillna(0) == 1
                                        )

                                    matching = training_df[symptom_mask]
                                    matching_rows = len(matching)
                                    predicted_class_matches = int(
                                        matching["prognosis"]
                                        .astype(str).str.strip()
                                        .eq(prediction).sum()
                                    )
                            except Exception:
                                training_check_available = False

                        # Conservative UI gate: never present a disease label as
                        # a supported result if the original training data has no
                        # row containing all confirmed symptoms for that class.
                        unsupported_by_training = (
                            training_check_available
                            and predicted_class_matches == 0
                        )
                        weak_model_support = (
                            top_score is not None and top_score < 0.50
                        )
                        close_competition = (
                            score_margin is not None and score_margin < 0.15
                        )
                        uncertain_result = (
                            unsupported_by_training
                            or weak_model_support
                            or close_competition
                        )

                        st.write("**Confirmed symptoms used for prediction:**")
                        st.write(
                            ", ".join(
                                symptom.replace("_", " ").title()
                                for symptom in sorted(all_symptoms)
                            )
                        )

                        st.divider()
                        st.subheader("Prediction Result")

                        if uncertain_result:
                            st.warning(
                                "Insufficient reliable support for a disease prediction. "
                                "Please do not treat the model's top-ranked class as a diagnosis."
                            )
                            st.write(
                                "**Model's top-ranked class (for technical review only):** "
                                + prediction
                            )
                        else:
                            st.info(
                                "**Model's top-ranked class (educational output):** "
                                + prediction
                            )

                        if top_score is not None:
                            st.caption(
                                f"Random Forest top score: {top_score * 100:.1f}% "
                                "(not a medical probability)."
                            )
                        if score_margin is not None:
                            st.caption(
                                f"Difference between top two model scores: "
                                f"{score_margin * 100:.1f} percentage points."
                            )

                        # Explainable AI: show model ranking and global feature
                        # importance for the symptoms selected in this case.
                        with st.expander("Explainable AI — understand the model output"):
                            st.markdown("**Top three model-ranked classes**")
                            if top_predictions:
                                ranking_df = pd.DataFrame(
                                    [
                                        {
                                            "Model-ranked class": label,
                                            "Relative model score": f"{score * 100:.1f}%"
                                        }
                                        for label, score in top_predictions
                                    ]
                                )
                                st.dataframe(
                                    ranking_df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                st.caption(
                                    "These are Random Forest score shares, not "
                                    "medical probabilities or disease risks."
                                )
                            else:
                                st.info(
                                    "This model does not expose class scores, so "
                                    "a ranked score explanation is unavailable."
                                )

                            if hasattr(model, "feature_importances_"):
                                importance_values = list(model.feature_importances_)
                                model_features = list(
                                    getattr(model, "feature_names_in_", features)
                                )
                                importance_map = {
                                    str(name): float(value)
                                    for name, value in zip(
                                        model_features, importance_values
                                    )
                                }
                                selected_importance = [
                                    {
                                        "Confirmed symptom": symptom.replace(
                                            "_", " "
                                        ).title(),
                                        "Global model importance": (
                                            importance_map.get(symptom, 0.0) * 100
                                        )
                                    }
                                    for symptom in sorted(all_symptoms)
                                ]
                                selected_importance.sort(
                                    key=lambda item: item["Global model importance"],
                                    reverse=True
                                )

                                st.markdown(
                                    "**Global importance of your confirmed symptoms**"
                                )
                                if selected_importance:
                                    importance_df = pd.DataFrame(
                                        selected_importance
                                    )
                                    importance_df[
                                        "Global model importance"
                                    ] = importance_df[
                                        "Global model importance"
                                    ].map(lambda value: f"{value:.3f}%")
                                    st.dataframe(
                                        importance_df,
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                else:
                                    st.info("No confirmed symptoms to explain.")

                                st.caption(
                                    "Feature importance describes how the trained "
                                    "Random Forest used features across its training "
                                    "process. It is not a case-specific causal "
                                    "explanation and does not mean a symptom caused "
                                    "the predicted class."
                                )

                                global_top = sorted(
                                    importance_map.items(),
                                    key=lambda item: item[1],
                                    reverse=True
                                )[:10]
                                with st.expander(
                                    "View the model's 10 most important features"
                                ):
                                    global_df = pd.DataFrame(
                                        [
                                            {
                                                "Feature": name.replace(
                                                    "_", " "
                                                ).title(),
                                                "Global importance": f"{value * 100:.3f}%"
                                            }
                                            for name, value in global_top
                                        ]
                                    )
                                    st.dataframe(
                                        global_df,
                                        use_container_width=True,
                                        hide_index=True
                                    )
                            else:
                                st.info(
                                    "This model does not expose built-in feature "
                                    "importance. A feature-level explanation is "
                                    "not available for this model."
                                )

                        if training_check_available:
                            st.caption(
                                f"Original training rows containing all selected "
                                f"symptoms: {matching_rows}; rows in the predicted "
                                f"class: {predicted_class_matches}."
                            )
                            if unsupported_by_training:
                                st.warning(
                                    "The original training dataset contains no row "
                                    "with all selected symptoms labelled as the "
                                    "model's top-ranked class."
                                )
                        else:
                            st.caption(
                                "Training-pattern cross-check was unavailable; "
                                "the system cannot verify support from the original dataset."
                            )

                        st.warning(
                            "Academic prototype only. Model scores and dataset "
                            "matches are not medical evidence, diagnosis, or treatment advice. "
                            "Consult a qualified healthcare professional for health concerns."
                        )

                        history_label = (
                            "Insufficient reliable support (top-ranked: "
                            + prediction + ")"
                            if uncertain_result
                            else prediction
                        )
                        database.save_prediction(
                            patient["patient_id"],
                            ", ".join(sorted(all_symptoms)),
                            history_label
                        )
                        st.success("Result saved to your patient history.")

                        for key in [
                            "pending_symptoms",
                            "pending_symptom_description",
                            "pending_symptom_uncertain",
                            "confirmed_symptoms"
                        ]:
                            st.session_state.pop(key, None)

    # ---------------- ML MODEL BENCHMARKING ----------------

    with benchmark_tab:

        st.subheader("Machine Learning Model Benchmarking")
        st.caption(
            "Compare standard classifiers using stratified cross-validation "
            "on the original Training.csv dataset."
        )

        st.warning(
            "This is an academic benchmark, not clinical validation. "
            "Results depend on the dataset and may be optimistic if records "
            "contain duplicated or highly similar symptom patterns."
        )

        if st.button(
            "Run 5-fold model comparison",
            type="primary",
            key="run_ml_benchmark"
        ):
            try:
                with st.spinner(
                    "Training and evaluating six models. This may take a few minutes..."
                ):
                    benchmark_df, row_count, feature_count, class_count, fold_count = (
                        run_model_benchmark()
                    )
                st.session_state["ml_benchmark_result"] = {
                    "table": benchmark_df,
                    "rows": row_count,
                    "features": feature_count,
                    "classes": class_count,
                    "folds": fold_count,
                }
            except Exception as exc:
                st.error(f"Benchmark could not be completed: {exc}")

        benchmark_result = st.session_state.get("ml_benchmark_result")
        if benchmark_result:
            st.markdown("#### Dataset overview")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Training records", benchmark_result["rows"])
            metric_cols[1].metric("Features", benchmark_result["features"])
            metric_cols[2].metric("Disease classes", benchmark_result["classes"])
            metric_cols[3].metric("CV folds", benchmark_result["folds"])

            st.markdown("#### Cross-validation results")
            display_df = benchmark_result["table"].copy()
            metric_columns = [
                "Accuracy", "Precision (macro)", "Recall (macro)",
                "F1-score (macro)", "F1 std."
            ]
            for column in metric_columns:
                display_df[column] = display_df[column].map(
                    lambda value: f"{value * 100:.2f}%"
                )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            chart_df = benchmark_result["table"].set_index("Model")[
                ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]
            ]
            st.markdown("#### Visual model comparison")
            st.bar_chart(chart_df)

            best_model = benchmark_result["table"].iloc[0]
            st.info(
                f"Highest macro F1 in this run: {best_model['Model']} "
                f"({best_model['F1-score (macro)'] * 100:.2f}%). "
                "This does not automatically mean it is the best model for "
                "real-world medical use."
            )

            st.caption(
                "The benchmark trains fresh baseline models using stratified "
                "cross-validation. It does not replace your deployed Random "
                "Forest model. The separate Testing.csv file is not used in "
                "this cross-validation comparison."
            )
        else:
            st.info(
                "Click the button above to run the benchmark. Results are "
                "cached for this app session to avoid unnecessary repeated training."
            )

    # ---------------- AI HEALTH ASSISTANT ----------------

    with assistant_tab:

        st.subheader("AI Health Assistant")
        st.caption(
            "Get general health information, a simple summary of your question, "
            "home-care information and guidance on when to seek medical attention."
        )

        st.info(
            "Privacy reminder: Do not enter your name, email, patient ID, address, "
            "or other identifying information. Your question and limited dashboard "
            "context are sent to OpenRouter, an external AI service."
        )

        response_language = st.selectbox(
            "Response language",
            ["Match my message", "English", "Hindi", "Hinglish"],
            key="assistant_response_language"
        )

        st.caption(
            "The assistant provides educational information only. It cannot diagnose, "
            "prescribe medicines or replace a qualified healthcare professional."
        )

        if "health_assistant_messages" not in st.session_state:
            st.session_state.health_assistant_messages = []

        for message in st.session_state.health_assistant_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input(
            "Ask a general health question (English, Hindi or Hinglish)..."
        )

        if user_question:
            st.session_state.health_assistant_messages.append(
                {"role": "user", "content": user_question}
            )

            with st.chat_message("user"):
                st.markdown(user_question)

            if not os.getenv("OPENROUTER_API_KEY"):
                assistant_reply = (
                    "OpenRouter API key is not configured. Set "
                    "OPENROUTER_API_KEY in the CMD window used to launch Streamlit."
                )
                with st.chat_message("assistant"):
                    st.error(assistant_reply)
            else:
                recent_health = database.get_health_history(patient["patient_id"])
                if recent_health:
                    latest_health = recent_health[0]
                    health_context = (
                        "Latest manually entered readings (not live sensor data): "
                        f"heart rate {latest_health.get('heart_rate')} BPM; "
                        f"blood pressure {latest_health.get('systolic_bp')}/"
                        f"{latest_health.get('diastolic_bp')} mmHg; "
                        f"SpO2 {latest_health.get('spo2')}%; "
                        f"temperature {latest_health.get('temperature')} °C; "
                        f"recorded at {latest_health.get('recorded_at')}."
                    )
                else:
                    health_context = "No saved health readings are available."

                recent_predictions = database.get_prediction_history(
                    patient["patient_id"]
                )
                if recent_predictions:
                    prediction_lines = []
                    for item in recent_predictions[:3]:
                        prediction_lines.append(
                            f"Symptoms entered: {item.get('symptoms', 'Not available')}; "
                            f"educational ML output: {item.get('predicted_class', 'Not available')}; "
                            f"recorded at: {item.get('recorded_at', 'Not available')}"
                        )
                    prediction_context = "Recent educational prediction history: " + " | ".join(prediction_lines)
                else:
                    prediction_context = "No saved model predictions are available."

                language_instruction = {
                    "Match my message": "Reply in the same language and style as the user's latest message.",
                    "English": "Reply in simple English.",
                    "Hindi": "Reply in simple Hindi using Devanagari script.",
                    "Hinglish": "Reply in simple conversational Hinglish using Latin script."
                }[response_language]

                system_prompt = f"""
You are an educational AI Health Assistant in a college academic prototype.
{language_instruction}

For each health-related question, use these headings when relevant:
1. **Understanding your concern** - briefly restate the question or symptoms without adding symptoms.
2. **General information** - explain possible general meanings; do not identify a definite disease.
3. **General self-care** - give low-risk, non-medicine guidance only when appropriate. Do not recommend prescription or OTC medicines, dosages, or treatment plans.
4. **When to seek medical help** - explain when the user should contact a qualified healthcare professional.
5. **Urgent warning** - if the user describes chest pain, severe breathing difficulty, fainting/unconsciousness, stroke signs, severe allergic reaction, heavy bleeding, or another immediate danger, tell them to seek local emergency medical help now. Do not delay urgent care for further chat.

Safety rules:
- Never diagnose, prescribe, recommend medicine dosages, or claim certainty.
- Do not treat the project's machine-learning output as a diagnosis or medical evidence.
- Dashboard readings are manually entered and may be inaccurate; do not infer a diagnosis from them.
- Explain that a single reading cannot establish a medical condition.
- Do not falsely reassure or say a person is safe based only on this limited context.
- Do not request personally identifying information.
- Keep the response clear, concise and practical. If the question is not medical, answer normally and briefly.
- This is an academic prototype, not a medical device.
"""

                context_message = (
                    "Limited context from the patient's local dashboard. Do not repeat "
                    "identifying details; none are supplied. "
                    + health_context + " " + prediction_context
                )

                try:
                    client = OpenAI(
                        api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url="https://openrouter.ai/api/v1"
                    )
                    response = client.chat.completions.create(
                        model="openrouter/free",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "system", "content": context_message},
                            *[
                                {"role": item["role"], "content": item["content"]}
                                for item in st.session_state.health_assistant_messages[-10:]
                            ]
                        ],
                        temperature=0.3
                    )
                    assistant_reply = (
                        response.choices[0].message.content
                        or "I could not generate a response. Please try again."
                    )
                    with st.chat_message("assistant"):
                        st.markdown(assistant_reply)

                except Exception as exc:
                    assistant_reply = (
                        "The AI service could not respond right now. Please try again "
                        "later. Technical detail: " + str(exc)
                    )
                    with st.chat_message("assistant"):
                        st.error(assistant_reply)

            st.session_state.health_assistant_messages.append(
                {"role": "assistant", "content": assistant_reply}
            )

        clear_col, info_col = st.columns([1, 3])
        with clear_col:
            if st.button("Clear AI chat", key="clear_health_assistant"):
                st.session_state.health_assistant_messages = []
                st.rerun()
        with info_col:
            st.caption(
                "Avoid sharing identifiable or sensitive medical information. "
                "For emergencies, contact local emergency services immediately."
            )

    # ---------------- PREDICTION HISTORY ----------------

    with history_tab:

        st.subheader("Your Previous Predictions")

        history = database.get_prediction_history(
            patient["patient_id"]
        )

        if history:

            history_df = pd.DataFrame(history)

            history_df = history_df.rename(columns={
                "predicted_class": "Predicted Result",
                "symptoms": "Symptoms",
                "recorded_at": "Date and Time"
            })

            st.dataframe(
                history_df[
                    ["Date and Time", "Symptoms", "Predicted Result"]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No prediction history yet. Your previous predictions "
                "will appear here."
            )



    # ---------------- PDF HEALTH REPORT ----------------
    with report_tab:
        st.subheader("Downloadable Patient Health Report")
        st.caption(
            "Create a PDF summary of your profile, manually entered health "
            "readings and saved educational model predictions."
        )
        report_health = database.get_health_history(patient["patient_id"])
        report_predictions = database.get_prediction_history(patient["patient_id"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Saved health readings", len(report_health))
        col2.metric("Saved predictions", len(report_predictions))
        col3.metric("Readings included", min(len(report_health), 30))

        st.info(
            "The report includes up to the latest 30 health readings and 20 "
            "prediction records. It contains sensitive personal information; "
            "download and share it only with the patient's permission."
        )

        if st.button("Generate PDF Health Report", type="primary"):
            try:
                pdf_bytes = build_patient_health_pdf(
                    patient,
                    report_health,
                    report_predictions
                )
                st.session_state["patient_health_pdf"] = pdf_bytes
                st.success("Your PDF report is ready.")
            except Exception as exc:
                st.error(f"Could not generate the PDF report: {exc}")

        if st.session_state.get("patient_health_pdf"):
            patient_code = re.sub(
                r"[^A-Za-z0-9_-]",
                "_",
                str(patient.get("patient_id", "patient"))
            )
            st.download_button(
                label="Download Patient Health Report (PDF)",
                data=st.session_state["patient_health_pdf"],
                file_name=f"health_report_{patient_code}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# ---------------- APPLICATION ENTRY ----------------

if st.session_state.patient is None:
    authentication_page()
else:
    patient_dashboard()

st.divider()

st.caption(
    "Major Project | AI-Based Smart Patient Health Monitoring "
    "and Early Disease Prediction System"
)
