import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="MedPredict", page_icon="🩺", layout="centered")

# Placeholder logo
st.image("https://via.placeholder.com/150x60?text=MedPredict+Logo", width=150)

# Title
st.title("🩺 Diabetes Risk Prediction")
st.markdown("Enter patient details below to predict diabetes risk.")

# Load trained model
if not os.path.exists("diabetes_model.pkl"):
    st.error("Model file not found. Please upload or train your model first.")
    st.stop()

model = joblib.load("diabetes_model.pkl")
st.success("📦 Model loaded successfully!")

# Input form
with st.form("prediction_form"):
    st.subheader("Patient Details")
    Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    Glucose = st.number_input("Glucose Level", 0, 300, 120)
    BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
    Insulin = st.number_input("Insulin Level", 0, 900, 80)
    BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    Age = st.number_input("Age", 1, 120, 30)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = model.predict(input_data)[0]
    result = "✅ No Diabetes" if prediction == 0 else "⚠️ Diabetes Detected"
    st.subheader("Result")
    st.success(f"Prediction: {result}")

# Upload CSV
st.markdown("---")
st.subheader("📂 Predict from CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(col in data.columns for col in expected_cols):
            st.error("CSV must contain: " + ", ".join(expected_cols))
        else:
            predictions = model.predict(data[expected_cols])
            data['Prediction'] = ["✅ No Diabetes" if p == 0 else "⚠️ Diabetes Detected" for p in predictions]
            st.write("🧾 Prediction Results")
            st.dataframe(data)
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Prediction Results", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")

# Model comparison
st.markdown("---")
st.subheader("📊 Model Accuracy Comparison")

model_choice = st.selectbox("Choose a model to evaluate", ["Random Forest", "Logistic Regression", "K-Nearest Neighbors"])

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

accuracies = {}

if model_choice == "Random Forest":
    clf = RandomForestClassifier()
elif model_choice == "Logistic Regression":
    clf = LogisticRegression(max_iter=1000)
else:
    clf = KNeighborsClassifier()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracies[model_choice] = acc

# Add all 3 models to bar chart
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier()
}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracies[name] = accuracy_score(y_test, pred)

fig, ax = plt.subplots()
ax.bar(accuracies.keys(), accuracies.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size: 14px;">
💡 Developed by <b>Kotha Mohan Krishna</b> | 📧 <a href="mailto:alwaysmohankrishnan@gmail.com">Contact</a>  
</div>
""", unsafe_allow_html=True)
