# streamlit_stomach_disease.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stomach Disease Predictor", layout="wide")

# ======================================================
# 1️⃣ Load Datasets
# ======================================================
@st.cache_data
def load_data():
    df_symptoms = pd.read_csv("stomach_disease_dataset.csv")
    df_doctors = pd.read_csv("doctor_dataset.csv")
    return df_symptoms, df_doctors

df_symptoms, df_doctors = load_data()

# ======================================================
# 2️⃣ Feature Encoding
# ======================================================
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_symptoms['symptom_list'])
y = df_symptoms['Disease']

# ======================================================
# 3️⃣ Train Ensemble Model
# ======================================================
rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=150, max_depth=8, learning_rate=0.1)

ensemble_model = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft'
)
ensemble_model.fit(X, y)

# ======================================================
# 4️⃣ Prediction & Doctor Recommendation Functions
# ======================================================
def predict_disease(user_symptoms, top_k=3):
    user_symptoms = [s.lower().strip() for s in user_symptoms]
    unknown = [s for s in user_symptoms if s not in mlb.classes_]
    if unknown:
        st.error(f"Invalid symptoms: {unknown}. Please choose from the available symptoms.")
        return []
    
    input_encoded = mlb.transform([user_symptoms])
    proba = ensemble_model.predict_proba(input_encoded)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]
    return [(ensemble_model.classes_[idx], proba[idx]) for idx in top_idx]

def recommend_doctor(disease):
    matching_doctors = df_doctors[df_doctors['Disease'].str.lower() == disease.lower()]
    if matching_doctors.empty:
        return None
    return matching_doctors.sample(1).iloc[0]

# ======================================================
# 5️⃣ Sidebar Inputs
# ======================================================
st.sidebar.header("Enter Symptoms")
selected_symptoms = st.sidebar.multiselect(
    "Choose symptoms from the list:",
    options=sorted(mlb.classes_)
)

top_k = st.sidebar.slider("Number of top predictions", 1, 5, 3)

if st.sidebar.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom!")
    else:
        # Predict
        predictions = predict_disease(selected_symptoms, top_k=top_k)
        if predictions:
            top_disease = predictions[0][0]
            st.subheader("🧠 Prediction Summary")
            st.write(f"**Entered Symptoms:** {', '.join(selected_symptoms)}")
            
            st.write("**Top Predicted Diseases:**")
            for disease, conf in predictions:
                st.write(f"- {disease} (Confidence: {conf*100:.2f}%)")

            # Doctor recommendation
            doctor = recommend_doctor(top_disease)
            if doctor is not None:
                st.write("**👨‍⚕️ Recommended Doctor:**")
                st.write(f"- Name: {doctor['Doctor_Name']}")
                st.write(f"- Specialization: {doctor['Doctor_Specialization']}")
                st.write(f"- Contact: {doctor['Doctor_Contact']}")
            else:
                st.write(f"⚠️ No doctor found for {top_disease}")

            # ======================================================
            # SHAP Explainability for single prediction
            # ======================================================
            rf_model = ensemble_model.named_estimators_['rf']
            explainer = shap.TreeExplainer(rf_model)
            input_encoded = mlb.transform([selected_symptoms])
            shap_values = explainer.shap_values(input_encoded)

            st.subheader("📊 SHAP Feature Contribution")
            shap.initjs()
            # Render force plot in Streamlit
            st_shap = st.pyplot(shap.force_plot(
                explainer.expected_value[np.argmax(rf_model.predict_proba(input_encoded))],
                shap_values[np.argmax(rf_model.predict_proba(input_encoded))],
                input_encoded,
                feature_names=mlb.classes_,
                matplotlib=True
            ))

# ======================================================
# 6️⃣ Global Visualization: Top 10 Symptoms
# ======================================================
st.subheader("Top 10 Most Frequent Symptoms in Dataset")
all_symptoms = [sym for sublist in df_symptoms['symptom_list'] for sym in sublist]
symptom_counts = Counter(all_symptoms)
top_symptoms = dict(symptom_counts.most_common(10))

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=list(top_symptoms.values()), y=list(top_symptoms.keys()), palette="viridis", ax=ax)
ax.set_xlabel("Frequency")
ax.set_ylabel("Symptoms")
ax.set_title("Top 10 Symptoms")
st.pyplot(fig)
