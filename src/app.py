import streamlit as st
import pandas as pd
import numpy as np
from audit_logic import ModelAuditor

st.set_page_config(page_title="ML Integrity Auditor", layout="wide")

st.title("ML Integrity & Leakage Auditor")
st.markdown("""
**Upload your dataset.** This tool will train a forensic model on your data to detect if any feature is "too good to be true" (Data Leakage).
""")

# --- SIDEBAR: INPUT ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

target_col = None
df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Loaded: {len(df)} rows, {len(df.columns)} cols")
        
        # Select Target Column
        st.sidebar.header("2. Configuration")
        all_cols = df.columns.tolist()
        target_col = st.sidebar.selectbox("Select Target Column (What you are predicting):", all_cols)
        
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

# --- MAIN: AUDIT ---
st.header("3. Audit Results")

if df is not None and target_col is not None:
    if st.button("Run Forensic Audit"):
        try:
            auditor = ModelAuditor(df, target_col)
            
            # 1. Train
            with st.status("Training Forensic Model...", expanded=True) as status:
                acc = auditor.train_forensic_model()
                status.update(label=f"Forensic Model Accuracy: {acc:.4f}", state="complete")
            
            # 2. SHAP
            with st.spinner("Running Game Theoretic Analysis (SHAP)..."):
                auditor.analyze_with_shap()
            
            # 3. Visualization
            st.subheader("Feature Dominance Report")
            
            # --- FIX: FORCE FLATTENING ---
            # Calculate Mean Absolute SHAP
            mean_shap = np.abs(auditor.shap_values).mean(axis=0)
            
            # Force it to be a 1D array (The fix for your error)
            mean_shap = np.array(mean_shap).flatten()
#
            # Double check shapes match before creating DataFrame
            if len(mean_shap) != len(auditor.X_test_sample.columns):
                st.error(f"Shape Mismatch: Features ({len(auditor.X_test_sample.columns)}) vs SHAP ({len(mean_shap)})")
                st.stop()
#
            # Create Dataframe
            feature_importance = pd.DataFrame({
                'feature': auditor.X_test_sample.columns,
                'importance': mean_shap
            }).sort_values(by='importance', ascending=False)
            # Normalize
            total = feature_importance['importance'].sum()
            feature_importance['percent'] = (feature_importance['importance'] / total) * 100
            
            # Display Top 10
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.bar_chart(feature_importance.set_index('feature')['percent'].head(10))
                
            with col2:
                st.write("Top Risk Factors:")
                st.dataframe(feature_importance[['feature', 'percent']].head(5).style.format({'percent': '{:.1f}%'}))
            
            # 4. Final Verdict Logic
            top_feature = feature_importance.iloc[0]
            
            if acc > 0.98 and top_feature['percent'] > 20:
                 st.error(f"🚨 CRITICAL LEAKAGE DETECTED: Feature '{top_feature['feature']}' is suspicious. It explains {top_feature['percent']:.1f}% of the model alone.")
            elif acc > 0.90 and top_feature['percent'] > 40:
                 st.warning(f"HIGH RISK: Feature '{top_feature['feature']}' is extremely dominant. Verify it's not a proxy for the target.")
            else:
                 st.success("Data looks healthy. No obvious leakage detected.")
                 
        except Exception as e:
            st.error(f"Audit Failed: {e}")
else:
    st.info("👈 Please upload a CSV file in the sidebar to begin.")