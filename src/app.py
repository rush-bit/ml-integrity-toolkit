import streamlit as st
import pandas as pd
import numpy as np
from leakage_injector import LeakageInjector
from audit_logic import ModelAuditor

# Page Config
st.set_page_config(page_title="ML Integrity Auditor", layout="wide")

st.title("🛡️ ML Integrity & Leakage Auditor")
st.markdown("""
**Objective:** Detect hidden data leakage in tabular models before deployment.
This tool simulates leakage injection and then uses **SHAP (Game Theory)** to audit the model.
""")

# Sidebar for controls
st.sidebar.header("1. Data Simulation")
# CHANGED: Default noise to 0.01 to make the leak VERY obvious for testing
noise_level = st.sidebar.slider("Injection Noise Level (Lower = Harder to detect)", 0.0, 0.5, 0.01)

if st.sidebar.button("Generate & Corrupt Data"):
    with st.spinner("Injecting Leakage..."):
        injector = LeakageInjector()
        df = injector.load_data()
        # Inject leakage based on slider
        df = injector.inject_target_leakage(df, noise_level=noise_level)
        df.to_csv("data/leaky_data.csv", index=False)
    st.sidebar.success(f"✅ Data Generated with Noise Level: {noise_level}")

# Main Area
st.header("2. Model Audit Results")

if st.button("Run Audit System"):
    try:
        auditor = ModelAuditor("data/leaky_data.csv")
        
        # 1. Train Model
        with st.status("Training Forensic Model...", expanded=True) as status:
            model = auditor.train_baseline_model()
            acc = model.score(auditor.X_test, pd.read_csv("data/leaky_data.csv")['target'].iloc[auditor.X_test.index])
            status.update(label=f"Model Trained! Accuracy: {acc:.4f}", state="complete")
        
        # 2. Calculate SHAP
        with st.spinner("Calculating SHAP Values..."):
            auditor.analyze_with_shap()
        
        # 3. Process Results
        st.subheader("Feature Importance (SHAP)")
        
        # Force 1D array for SHAP values
        mean_shap = np.abs(auditor.shap_values).mean(axis=0)
        mean_shap = np.array(mean_shap).flatten()
        
        feature_importance = pd.DataFrame({
            'feature': auditor.X_test.columns,
            'importance': mean_shap
        }).sort_values(by='importance', ascending=False)
        
        # Normalize to percentage
        total_importance = feature_importance['importance'].sum()
        feature_importance['importance_percent'] = (feature_importance['importance'] / total_importance) * 100
        
        # Show the top 5 raw numbers so you can debug
        st.write("Top 5 Drivers of Prediction:")
        st.dataframe(feature_importance.head(5).style.format({'importance_percent': '{:.2f}%'}))
        
        # Plot
        st.bar_chart(feature_importance.set_index('feature')['importance'])
        
        # 4. The Verdict (New Logic)
        top_1_score = feature_importance.iloc[0]['importance_percent']
        top_2_score = feature_importance.iloc[0]['importance_percent'] + feature_importance.iloc[1]['importance_percent']
        
        # Logic: If Top 1 is > 15% OR Top 2 combined are > 25% AND Accuracy is super high
        if acc > 0.98 and (top_1_score > 15 or top_2_score > 25):
            st.error(f"🚨 LEAKAGE DETECTED! The top features explain {top_2_score:.1f}% of the model. Accuracy is {acc:.4f}. This is suspicious.")
        elif top_1_score > 40:
             st.warning(f"⚠️ High Dependence: Feature '{feature_importance.iloc[0]['feature']}' drives {top_1_score:.1f}% of the model.")
        else:
            st.success("✅ Model looks robust. Importance is distributed.")
            
    except FileNotFoundError:
        st.warning("⚠️ Please generate data in the sidebar first!")
    except Exception as e:
        st.error(f"An error occurred: {e}")