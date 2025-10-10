# ======================================================
# 🌐 Stomach Disease Prediction App (Streamlit Version)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

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
# 2️⃣ Model Training & Encoding
# ======================================================
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_symptoms['symptom_list'])
y = df_symptoms['Disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
xgb = XGBClassifier(
    use_label_encoder=False, eval_metric='mlogloss',
    n_estimators=150, max_depth=8, learning_rate=0.1
)

ensemble_model = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)

# ======================================================
# 3️⃣ Streamlit Page Setup
# ======================================================
st.set_page_config(page_title="🩺 Stomach Disease Predictor", layout="wide")
st.title("🧠 Stomach Disease Prediction System")
st.markdown("### Predict diseases from symptoms and get doctor recommendations")

# ======================================================
# 4️⃣ Model Performance
# ======================================================
st.subheader("📊 Model Performance")

y_pred = ensemble_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"✅ **Accuracy:** {acc*100:.2f}%")

with st.expander("See classification report"):
    st.text(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            xticklabels=ensemble_model.classes_, yticklabels=ensemble_model.classes_,
            cmap='crest', ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# ======================================================
# 5️⃣ SHAP Global Explainability
# ======================================================
st.subheader("🔍 SHAP Feature Importance (Explainability)")

rf_model = ensemble_model.named_estimators_['rf']
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, features=X_test, feature_names=mlb.classes_,
                  show=False, plot_type="bar", color_bar=True)
plt.title("Feature Importance - Most Contributing Symptoms", fontsize=14)
st.pyplot(fig)

# ======================================================
# 6️⃣ Disease Prediction and Doctor Recommendation
# ======================================================
st.subheader("🩺 Disease Prediction")

symptom_input = st.multiselect(
    "Select your symptoms:",
    sorted(mlb.classes_)
)

if st.button("🔮 Predict Disease"):
    if not symptom_input:
        st.warning("Please select at least one symptom.")
    else:
        user_symptoms = [s.lower().strip() for s in symptom_input]
        input_encoded = mlb.transform([user_symptoms])
        proba = ensemble_model.predict_proba(input_encoded)[0]
        top_idx = np.argsort(proba)[::-1][:3]

        st.markdown("### 🧠 Prediction Results:")
        for idx in top_idx:
            disease = ensemble_model.classes_[idx]
            confidence = proba[idx]
            st.write(f"**{disease}** — Confidence: {confidence*100:.2f}%")

        top_disease = ensemble_model.classes_[top_idx[0]]
        st.success(f"✅ Predicted Disease: **{top_disease}**")

        # Doctor Recommendation
        matching_doctors = df_doctors[df_doctors['Disease'].str.lower() == top_disease.lower()]
        if not matching_doctors.empty:
            doctor = matching_doctors.sample(1).iloc[0]
            st.markdown("### 👨‍⚕️ Recommended Doctor")
            st.write(f"**Name:** {doctor['Doctor_Name']}")
            st.write(f"**Specialization:** {doctor['Doctor_Specialization']}")
            st.write(f"**Contact:** {doctor['Doctor_Contact']}")
        else:
            st.info(f"No specific doctor found for **{top_disease}**.")

# ======================================================
# 7️⃣ SHAP Explanation for Single Prediction
# ======================================================
st.subheader("🔎 Explain Model Prediction (SHAP)")

if symptom_input:
    input_encoded = mlb.transform([symptom_input])
    shap_values_input = explainer.shap_values(input_encoded)
    top_idx = np.argmax(rf_model.predict_proba(input_encoded))

    st.write("Feature importance for your selected symptoms:")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values_input[top_idx], input_encoded,
                      feature_names=mlb.classes_, show=False, plot_type="bar")
    st.pyplot(fig)

# ======================================================
# 8️⃣ Visualization of Dataset Insights
# ======================================================
st.subheader("📈 Dataset Insights")

all_symptoms = [sym for sublist in df_symptoms['symptom_list'] for sym in sublist]
symptom_counts = Counter(all_symptoms)
top_symptoms = dict(symptom_counts.most_common(10))

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=list(top_symptoms.values()), y=list(top_symptoms.keys()), palette="viridis", ax=ax)
ax.set_title("Top 10 Most Frequent Symptoms", fontsize=14)
st.pyplot(fig)

st.caption("Built with ❤️ using Streamlit, Scikit-learn, XGBoost, and SHAP.")
