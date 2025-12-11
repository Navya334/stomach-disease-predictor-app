# ================================
# 1️⃣ Import Libraries
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# 2️⃣ Load Model, Encoder, and Doctor Dataset
# ================================
model = joblib.load("stomach_disease_model.pkl")
mlb = joblib.load("symptom_encoder.pkl")
df_doctors = pd.read_csv("doctor_dataset.csv")

# ================================
# 3️⃣ Streamlit Page Setup
# ================================
st.set_page_config(page_title="Stomach Disease Prediction", layout="centered")
st.title("Stomach Disease Prediction System")
st.write("""
This system predicts possible stomach-related diseases based on your entered symptoms
and recommends the most suitable doctor for consultation.
""")

# ================================
# 4️⃣ User Input
# ================================
st.subheader("Select Your Symptoms (Max 6)")
selected_symptoms = st.multiselect(
    "Choose symptoms you are experiencing:",
    options=sorted(mlb.classes_),
    help="You can select up to 6 symptoms only."
)

# ================================
# 5️⃣ Check symptom limit and disable Predict button
# ================================
disable_predict = False
if len(selected_symptoms) > 6:
    st.error("🚫 You have exceeded the limit! Please select up to 6 symptoms only.")
    disable_predict = True
elif len(selected_symptoms) == 0:
    st.warning("⚠️ Please select at least one symptom to proceed.")
    disable_predict = True

# ================================
# 6️⃣ Prediction Button
# ================================
if st.button("Predict Disease", disabled=disable_predict):

    # Encode input
    input_encoded = mlb.transform([selected_symptoms])

    # Predict probabilities
    proba = model.predict_proba(input_encoded)[0]
    top_idx = np.argsort(proba)[::-1][:3]

    # Display top predictions
    st.markdown("---")
    st.subheader("Prediction Results")
    for idx in top_idx:
        disease = model.classes_[idx]
        confidence = proba[idx] * 100
        st.write(f"**{disease}** — Confidence: `{confidence:.2f}%`")

    # Pick top prediction
    top_disease = model.classes_[top_idx[0]]

    # Recommend doctor
    st.markdown("---")
    st.subheader("Recommended Doctor")
    matching_doctors = df_doctors[df_doctors['Disease'].str.lower() == top_disease.lower()]

    if not matching_doctors.empty:
        doctor = matching_doctors.sample(1).iloc[0]
        st.success(f"**Doctor Name:** {doctor['Doctor_Name']}")
        st.write(f"**Specialization:** {doctor['Doctor_Specialization']}")
        st.write(f"**Contact:** {doctor['Doctor_Contact']}")
    else:
        st.warning(f"No doctor found for **{top_disease}** in the dataset.")

    st.markdown("---")
    st.info(f"Entered Symptoms: {', '.join(selected_symptoms)}")

# ================================
# 7️⃣ Footer
# ================================
st.markdown("""
---
 **Developed by:** AI-based Stomach Disease Analysis Team  
*Predict with confidence and consult the right doctor!*
""")

