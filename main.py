import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("Randomforestclassifier.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("standard.pkl")

model = load_model()
scaler = load_scaler()

st.title("🩺 Health Issue Prediction App")
st.write("Predict your potential health issue based on your body vitals and lifestyle factors.")

age = st.number_input("Age", min_value=1, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=22.5)
bp = st.selectbox("Blood Pressure Level", ["Normal", "High"])
chol = st.slider("Cholesterol (mg/dL)", 100, 400, 180)
glucose = st.slider("Glucose Level (mg/dL)", 70, 300, 90)
heart_rate = st.slider("Heart Rate (bpm)", 50, 120, 75)
smoking = st.selectbox("Smoking Habit", ["No", "Yes"])
alcohol = st.selectbox("Alcohol Intake", ["No", "Yes"])
exercise = st.selectbox("Exercise Frequency (per week)", [0, 1, 2, 3, 4, 5, 6, 7])
sleep = st.slider("Sleep Hours (per night)", 3, 12, 8)


gender_map = {"Male": 1, "Female": 0}
bp_map = {"Normal": 0, "High": 1}
smoking_map = {"No": 0, "Yes": 1}
alcohol_map = {"No": 0, "Yes": 1}


input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_map[gender]],
    'BMI': [bmi],
    'Blood_Pressure': [bp_map[bp]],
    'Cholesterol': [chol],
    'Glucose': [glucose],
    'Heart_Rate': [heart_rate],
    'Smoking': [smoking_map[smoking]],
    'Alcohol_Intake': [alcohol_map[alcohol]],
    'Exercise_Frequency': [exercise],
    'Sleep_Hours': [sleep]
})


scale_features = ["Age", "BMI", "Blood_Pressure", "Cholesterol", "Glucose", "Heart_Rate"]
input_data[scale_features] = scaler.transform(input_data[scale_features])


expected_features = [
    "Age","Gender","BMI","Blood_Pressure","Cholesterol",
    "Glucose","Heart_Rate","Smoking","Alcohol_Intake",
    "Exercise_Frequency","Sleep_Hours"
]
input_data = input_data[expected_features]


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]

    issue_map = {
        1: "🟢 No Health Issue — You're Healthy!",
        0: "🩸 Possible Diabetes — Maintain diet & check sugar levels.",
        2: "❤ Possible Heart Disease — Exercise and manage stress.",
        3: "🧬 Possible Liver Problem — Reduce alcohol/fat intake."
    }

    st.subheader("🔍 Prediction Result:")
    st.success(issue_map[prediction])

    st.write("### Prediction Confidence:")
    for i, prob in enumerate(probs):
        st.write(f"{issue_map[i]} → {prob*100:.2f}%")

    if prediction == 0:
        st.info("Tip: Keep up your healthy lifestyle — balanced diet, regular sleep, and exercise.")
    else:
        st.warning("Consider consulting a doctor for confirmation and advice.")