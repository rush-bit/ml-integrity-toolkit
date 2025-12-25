# 🛡️ ML Integrity & Audit Toolkit

![Demo Animation](assets/Untitled video - Made with Clipchamp (1).gif)(Untitled video - Made with Clipchamp.gif)

> **Automated forensic auditing for tabular machine learning models.**
> *Detects Target Leakage, Feature Dominance, and Overfitting before deployment.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![Status](https://img.shields.io/badge/Status-MVP-success)

## 🚨 The Problem
In real-world ML deployment, **Data Leakage** is the silent killer. Models often perform perfectly during training because they inadvertently "cheat" by accessing information that won't be available in production (e.g., future timestamps, proxy IDs). 

Standard validation metrics (Accuracy/AUC) fail to catch this because the leak exists in both Train and Test sets.

## 🛠️ The Solution
This toolkit provides a **forensic dashboard** that stresses the model using Game Theoretic explanations (SHAP) to identify features that are "too good to be true."

### Key Capabilities
* **Automated Leakage Injection:** Simulates real-world data corruption (Target Leakage & ID Proxy Leakage) to test audit capabilities.
* **SHAP-Based Forensics:** Uses Shapley values to detect non-linear feature dominance.
* **Heuristic Audit Engine:** Automatically flags models with >98% accuracy driven by a small subset of features (<25% of feature space).
* **Interactive Dashboard:** A Streamlit UI for non-technical stakeholders to visualize model trustworthiness.

## 📸 How It Works
1. **Upload Data:** Drag and drop any CSV dataset (Classification or Regression).
2. **Select Target:** Choose the column you want to predict.
3. **Forensic Audit:** The system trains a shadow model to detect patterns.
4. **Leakage Detection:** - **SHAP Analysis:** Calculates the marginal contribution of every feature.
   - **Red Flag Logic:** If a single feature drives >95% of accuracy alone, the system triggers a **CRITICAL LEAKAGE ALERT**.
   
## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/rush-bit/ml-integrity-toolkit.git](https://github.com/rush-bit/ml-integrity-toolkit.git)
cd ml-integrity-toolkit