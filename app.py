# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 1️⃣ Load Data
# ======================================================
@st.cache_data
def load_data():
    df_symptoms = pd.read_csv("stomach_disease_dataset.csv")
    df_doctors = pd.read_csv("doctor_dataset.csv")
    
    # Combine symptom columns
    symptom_cols = [col for col in df_symptoms.columns if 'Symptom' in col]
    df_symptoms['symptom_list'] = df_symptoms[symptom_cols].values.tolist()
    
    return df_symptoms, df_doctors

df_symptoms, df_doctors = load_data()

# ======================================================
# 2️⃣ Feature Encoding & Model Training
# ======================================================
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_symptoms['symptom_list'])
y = df_symptoms['Disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                    n_estimators=150, max_depth=8, learning_rate=0.1)

ensemble_model = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft'
)

ensemble_model.fit(X_train, y_train)

# ======================================================
# 3️⃣ Streamlit UI
# ======================================================
st.title(" Stomach Disease Prediction System")
st.markdown("### Predict stomach-related diseases and get doctor recommendations 🩺")

# Sidebar info
st.sidebar.header("🔍 About")
st.sidebar.info("This app uses **AI (Random Forest + XGBoost)** to predict diseases from symptoms and recommend doctors.")

# Display available symptoms
all_symptoms = sorted(set(sym for sublist in df_symptoms['symptom_list'] for sym in sublist))
selected_symptoms = st.multiselect(
    "Select your symptoms:", all_symptoms
)

# ======================================================
# 4️⃣ Prediction Function
# ======================================================
def predict_disease(user_symptoms):
    user_symptoms = [s.lower().strip() for s in user_symptoms]
    unknown = [s for s in user_symptoms if s not in mlb.classes_]
    if unknown:
        st.error(f"Invalid symptoms: {unknown}")
        return None

    input_encoded = mlb.transform([user_symptoms])
    proba = ensemble_model.predict_proba(input_encoded)[0]
    top_idx = np.argsort(proba)[::-1][:3]

    results = []
    for idx in top_idx:
        disease = ensemble_model.classes_[idx]
        confidence = proba[idx]
        results.append((disease, confidence))
    return results

def recommend_doctor(disease):
    match = df_doctors[df_doctors['Disease'].str.lower() == disease.lower()]
    if match.empty:
        return None
    return match.sample(1).iloc[0]

# ======================================================
# 5️⃣ Display Results
# ======================================================
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        predictions = predict_disease(selected_symptoms)
        if predictions:
            top_disease = predictions[0][0]
            st.success(f"### ✅ Predicted Disease: {top_disease}")
            st.write("**Confidence Levels:**")
            for dis, conf in predictions:
                st.write(f"- {dis}: {conf*100:.2f}%")

            # Doctor recommendation
            doctor = recommend_doctor(top_disease)
            if doctor is not None:
                st.subheader("👨‍⚕️ Recommended Doctor")
                st.write(f"**Name:** {doctor['Doctor_Name']}")
                st.write(f"**Specialization:** {doctor['Doctor_Specialization']}")
                st.write(f"**Contact:** {doctor['Doctor_Contact']}")
            else:
                st.warning(f"No doctor found for {top_disease} in dataset.")

# ======================================================
# 8️⃣ Symptom Frequency Visualization
# ======================================================
with st.expander("📈 Symptom Frequency Analysis"):
    from collections import Counter
    all_symptoms_flat = [sym for sublist in df_symptoms['symptom_list'] for sym in sublist]
    symptom_counts = Counter(all_symptoms_flat)
    top_symptoms = dict(symptom_counts.most_common(10))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=list(top_symptoms.values()), y=list(top_symptoms.keys()), palette="viridis", ax=ax)
    plt.title("Top 10 Most Frequent Symptoms")
    plt.xlabel("Frequency")
    plt.ylabel("Symptoms")
    st.pyplot(fig)
