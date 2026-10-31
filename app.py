
import json
import os
import re
import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

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

    profile_tab, monitoring_tab, prediction_tab, history_tab, assistant_tab = st.tabs([
        "Patient Profile",
        "Health Monitoring",
        "Disease Prediction",
        "Prediction History",
        "AI Health Assistant"
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
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(input_df)[0]
                            ranked_scores = sorted(
                                [float(value) for value in probabilities],
                                reverse=True
                            )
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

    # ---------------- AI HEALTH ASSISTANT ----------------

    with assistant_tab:

        st.subheader("AI Health Assistant")
        st.caption(
            "Ask general health questions or request simple explanations "
            "of your manually entered readings and previous model predictions."
        )

        st.info(
            "Do not enter your name, email, patient ID, or other identifying "
            "information. Messages are sent to an external AI provider."
        )

        if "health_assistant_messages" not in st.session_state:
            st.session_state.health_assistant_messages = []

        for message in st.session_state.health_assistant_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input(
            "Ask a general health question (English or Hinglish)..."
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
                health_context = "No saved health readings are available."
                recent_health = database.get_health_history(patient["patient_id"])
                if recent_health:
                    latest_health = recent_health[0]
                    health_context = (
                        "Latest manually entered readings: "
                        f"heart rate {latest_health['heart_rate']} BPM; "
                        f"blood pressure {latest_health['systolic_bp']}/"
                        f"{latest_health['diastolic_bp']} mmHg; "
                        f"SpO2 {latest_health['spo2']}%; "
                        f"temperature {latest_health['temperature']} °C."
                    )

                recent_predictions = database.get_prediction_history(
                    patient["patient_id"]
                )
                prediction_context = "No saved model predictions are available."
                if recent_predictions:
                    latest_prediction = recent_predictions[0]
                    prediction_context = (
                        "Most recent educational ML model output: "
                        f"{latest_prediction.get('predicted_class', 'Not available')}; "
                        f"symptoms entered: {latest_prediction.get('symptoms', 'Not available')}."
                    )

                system_prompt = """
You are an educational AI Health Assistant inside a college project.
Answer in clear, simple English or Hinglish matching the user's language.
You may explain general health concepts and supplied readings or educational
ML output, but never diagnose, prescribe medicines, recommend dosages, or
claim certainty. Do not treat the ML prediction as a diagnosis. Encourage a
qualified healthcare professional for personal medical concerns.
For severe or potentially life-threatening symptoms (such as chest pain,
severe breathing difficulty, fainting, stroke signs, or heavy bleeding), advise
urgent local emergency medical help immediately. Do not falsely reassure.
Keep answers concise and practical. Do not ask for personally identifying data.
This is an academic prototype, not a medical device.
"""

                context_message = (
                    "Context from local dashboard (manual data only): "
                    f"{health_context} {prediction_context}"
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
                        temperature=0.4
                    )
                    assistant_reply = (
                        response.choices[0].message.content
                        or "I could not generate a response. Please try again."
                    )
                    with st.chat_message("assistant"):
                        st.markdown(assistant_reply)

                except Exception as exc:
                    assistant_reply = (
                        "The AI service could not respond right now. "
                        "Please try again later. Error: " + str(exc)
                    )
                    with st.chat_message("assistant"):
                        st.error(assistant_reply)

            st.session_state.health_assistant_messages.append(
                {"role": "assistant", "content": assistant_reply}
            )

        if st.button("Clear AI chat", key="clear_health_assistant"):
            st.session_state.health_assistant_messages = []
            st.rerun()

        st.caption(
            "Avoid sharing identifiable or sensitive medical information. "
            "This assistant provides general educational information, not "
            "diagnosis or treatment."
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
