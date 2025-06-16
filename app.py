import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="🩺 Diabetes Risk Predictor",
    page_icon="🧬",
    layout="centered",
)

st.title("🩺 Diabetes Risk Prediction")

st.markdown("Enter patient details below to predict diabetes risk.")

# Load model
try:
    model = joblib.load("diabetes_model.pkl")
    st.success("📦 Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Input fields
with st.form("prediction_form"):
    st.subheader("Patient Details")
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    Age = st.number_input("Age", min_value=1, max_value=120, value=30)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    try:
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = model.predict(input_data)[0]
        result = "✅ No Diabetes" if prediction == 0 else "⚠️ Diabetes Detected"
        st.subheader("Result")
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# Optional: Expandable project info
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This app uses a machine learning model trained on the **Pima Indians Diabetes Dataset**  
    to predict the likelihood of a person having diabetes based on their medical input data.
    - Developed with `scikit-learn` and `Streamlit`
    - Input values are customizable and live
    - For educational/demonstration purposes only
    """)
st.markdown("---")
st.subheader("📂 Predict from CSV File")

uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV
        data = pd.read_csv(uploaded_file)

        st.write("📄 Uploaded Data Preview:")
        st.dataframe(data.head())

        # Check for required columns
        expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        if not all(col in data.columns for col in expected_cols):
            st.error("❌ CSV must contain columns: " + ", ".join(expected_cols))
        else:
            # Predict
            predictions = model.predict(data[expected_cols])
            data['Prediction'] = ["✅ No Diabetes" if p == 0 else "⚠️ Diabetes Detected" for p in predictions]

            st.write("🧾 Prediction Results:")
            st.dataframe(data)

            # Download button
            csv_download = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Prediction Results as CSV",
                data=csv_download,
                file_name="predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

