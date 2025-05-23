import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

# Load datasets
df_symptoms = pd.read_csv("stomach_disease_dataset.csv")
df_doctors = pd.read_csv("doctor_dataset.csv")

# Prepare data
symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6']
df_symptoms['symptom_list'] = df_symptoms[symptom_cols].values.tolist()

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_symptoms['symptom_list'])
y = df_symptoms['Disease']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# App UI
st.title("🩺 Stomach Disease Predictor")
st.markdown("**Select up to 6 symptoms**")

# Symptoms multiselect
symptoms_selected = st.multiselect("Choose your symptoms", sorted(mlb.classes_))

# Check selection limit
if len(symptoms_selected) > 6:
    st.error("You can only select up to 6 symptoms.")
    disable_predict = True
else:
    disable_predict = False

# Prediction logic
if st.button("Predict Disease", disabled=disable_predict):
    if not symptoms_selected:
        st.warning("Please select at least one symptom.")
    else:
        input_encoded = mlb.transform([symptoms_selected])
        predicted_disease = model.predict(input_encoded)[0]
        st.success(f"Predicted Disease: {predicted_disease}")

        matching_doc = df_doctors[df_doctors['Disease'] == predicted_disease]

        if matching_doc.empty:
            st.warning("No doctor information available for this disease.")
        else:
            doc = matching_doc.iloc[0]
            st.write(f"**Doctor Name:** {doc['Doctor_Name']}")
            st.write(f"**Specialization:** {doc['Doctor_Specialization']}")
            st.write(f"**Contact:** {doc['Doctor_Contact']}")
            st.warning("⚠This app is for educational use only. It does not give medical advice. Please consult a doctor for medical help.")
