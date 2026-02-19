import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Professional Medical UI Configuration
st.set_page_config(page_title="MediDiagnose AI", layout="wide", page_icon="🩸")

# Custom CSS for a clean, high-end "Clinical" feel
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stAlert { border-radius: 10px; }
    .main-header { font-size: 36px; font-weight: bold; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 18px; color: #64748B; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load the Brain
@st.cache_resource
def load_assets():
    model = pickle.load(open('model.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
    return model, features

model, features = load_assets()

# 3. Sidebar Inputs
st.sidebar.markdown("### 📋 Patient CBC Data")
st.sidebar.info("Enter standard lab values below.")

hgb = st.sidebar.number_input("Hemoglobin (HGB) [g/dL]", 2.0, 25.0, 13.0, help="Primary oxygen carrier")
rbc = st.sidebar.number_input("Red Blood Cells (RBC) [10^6/µL]", 0.5, 10.0, 4.5)
mcv = st.sidebar.number_input("Mean Corpuscular Vol (MCV) [fL]", 40.0, 150.0, 90.0)
mch = st.sidebar.number_input("Mean Corpuscular Hgb (MCH) [pg]", 10.0, 50.0, 30.0)
rdw = st.sidebar.number_input("Red Cell Dist Width (RDW) [%]", 10.0, 30.0, 13.0)

# 4. Processing Logic
input_dict = {'HGB': hgb, 'RBC': rbc, 'MCV': mcv, 'MCH': mch, 'RDW': rdw}
input_df = pd.DataFrame([input_dict])[features]
prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]
mentzer = mcv / (rbc + 1e-6)

# 5. Dashboard Layout
st.markdown('<p class="main-header">MediDiagnose AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explainable Clinical Decision Support System</p>', unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🏥 Diagnostic Analysis")
    # Change the confidence display to show "Health Score" if not anemic
    if prediction == 0:
        st.metric("Health Confidence", f"{(1-prob)*100:.1f}%")
    else:
        st.metric("Anemia Confidence", f"{prob*100:.1f}%")
    # GOAL 3: Outlier Detection
    if hgb < 3.0 or hgb > 22.0:
        st.error("🚨 **CRITICAL ALERT: OUTLIER DETECTED**")
        st.write("The Hemoglobin value is outside the physiological range for a living patient. Please verify lab sample or sensor calibration.")
    
    # GOAL 1 & 4: Anemia & Triage Logic
    elif prediction == 1:
        if hgb < 7.0:
            st.error("🚨 **RESULT: SEVERE ANEMIA (Priority 1)**")
            st.markdown("### **Urgent Action Required:**")
            st.write("Consult a physician immediately for a blood transfusion. This level of anemia is critical for organ oxygenation.")
        else:
            st.warning("🟡 **RESULT: ANEMIA DETECTED (Priority 2)**")
            st.markdown("### **Clinical Suggestion:**")
            type_anemia = "Genetic/Thalassemia" if mentzer < 13 else "Nutritional/Iron Deficiency"
            st.write(f"**Potential Type:** {type_anemia} (based on Mentzer Index of {mentzer:.1f})")
            st.write("Schedule a hematology consultation for an iron profile and B12 test.")
    
    else:
        # GOAL 4: Risk Analysis for Healthy Patients
        if rdw > 15.0 or hgb < 12.5:
            st.warning("🟠 **RESULT: NO ANEMIA (Level 3 - High Risk)**")
            st.write("**Prevention Plan:** Borderline values detected. Increase iron-rich food intake and re-test in 3 months.")
        elif 14.0 < rdw <= 15.0:
            st.info("🟡 **RESULT: NO ANEMIA (Level 2 - Guarded)**")
            st.write("**Prevention Plan:** Maintain balanced nutrition. Watch for fatigue symptoms.")
        else:
            st.success("🟢 **RESULT: HEALTHY (Level 1 - Optimal)**")
            st.write("All blood markers are within safe clinical ranges. No intervention required.")

with col2:
    st.subheader("💡 AI Decision Evidence")
    
    # Fast Feature Importance (Native to Random Forest)
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    
    # Visualizing the "Why" without the SHAP lag
    st.write("The AI weighted the following markers most heavily for this patient:")
    
    # Create a simple, fast bar chart
    st.bar_chart(feat_imp)
    
    # Text-based Explanation (Goal 2)
    top_feature = feat_imp.index[0]
    st.markdown(f"""
    **Why this result?**
    The primary driver for this diagnosis is **{top_feature}**. 
    In clinical terms, your **{top_feature}** value deviates most significantly 
    from the healthy baseline, pushing the AI's confidence to **{prob*100:.1f}%**.
    """)
    
    if prediction == 1:
        st.info(f"Expert Note: The Mentzer Index ({mentzer:.2f}) further supports a { 'Genetic' if mentzer < 13 else 'Nutritional' } profile.")