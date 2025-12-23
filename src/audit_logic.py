import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class ModelAuditor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.model = None
        self.X_test = None
        self.shap_values = None

    def train_baseline_model(self):
        """
        Trains a quick Random Forest. 
        If leakage exists, this model will have suspiciously high accuracy (near 100%).
        """
        print("Training Forensic Model...")
        
        # Prepare X and y
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        
        # Split (Standard 80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_test = X_test
        
        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Check accuracy
        acc = self.model.score(X_test, y_test)
        print(f"Model Accuracy: {acc:.4f} (If > 0.98, suspect leakage!)")
        return self.model

    def analyze_with_shap(self):
        """
        Calculates SHAP values. 
        Includes a fix for the 'Dimension Mismatch' error on newer SHAP versions.
        """
        print("Calculating SHAP Values (This might take a moment)...")
        
        explainer = shap.TreeExplainer(self.model)
        # check_additivity=False prevents errors on some noisy datasets
        self.shap_values = explainer.shap_values(self.X_test, check_additivity=False)
        
        # --- ROBUST SHAPE HANDLING START ---
        # Case A: It returns a list (Older SHAP). Index 1 = Positive Class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
            
        # Case B: It returns a 3D array (Newer SHAP). Shape: (Samples, Features, Classes)
        # We need to slice it to get only the Positive Class (Index 1 on the last axis)
        elif len(np.shape(self.shap_values)) == 3:
            self.shap_values = self.shap_values[:, :, 1]
        # --- ROBUST SHAPE HANDLING END ---

        # Final Safety Check: Verify it is now 2D (Samples, Features)
        if len(np.shape(self.shap_values)) != 2:
            raise ValueError(f"SHAP shape mismatch. Expected 2D array, got {np.shape(self.shap_values)}")

    def get_leakage_report(self):
        """
        Identifies features that have disproportionately high SHAP importance.
        """
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        feature_names = self.X_test.columns
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values(by='importance', ascending=False)
        
        total_importance = importance_df['importance'].sum()
        importance_df['importance_percent'] = (importance_df['importance'] / total_importance) * 100
        
        print("\nLEAKAGE AUDIT REPORT ")
        print("-" * 30)
        print(importance_df.head(5))
        
        # --- IMPROVED HEURISTIC ---
        top_feature = importance_df.iloc[0]
        
        # Check 1: Is the top feature suspiciously dominant? (Lowered to 20%)
        is_dominant = top_feature['importance_percent'] > 20
        
        # Check 2: Is the model "Too Good To Be True"?
        accuracy = self.model.score(self.X_test, pd.read_csv("data/leaky_data.csv")['target'].iloc[self.X_test.index])
        is_perfect = accuracy > 0.98

        if is_dominant and is_perfect:
            print(f"\nCRITICAL WARNING: Feature '{top_feature['feature']}' is driving {top_feature['importance_percent']:.1f}% of predictions.")
            print("VERDICT: HIGH PROBABILITY OF LEAKAGE (Perfect Accuracy + High Dependence)")
        elif is_dominant:
            print(f"\nWARNING: Feature '{top_feature['feature']}' is very strong ({top_feature['importance_percent']:.1f}%). check business logic.")
        else:
            print("\nModel looks balanced.")

if __name__ == "__main__":
    # Test on the LEAKY data we created
    auditor = ModelAuditor("data/leaky_data.csv")
    auditor.train_baseline_model()
    auditor.analyze_with_shap()
    auditor.get_leakage_report()