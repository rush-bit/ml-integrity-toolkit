import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ModelAuditor:
    def __init__(self, df, target_col):
        """
        Initializes the auditor with a user-provided dataframe and target column.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.model = None
        self.X_test = None
        self.shap_values = None
        self.problem_type = "classification" # Default

    def preprocess_data(self):
        """
        Basic preprocessing to handle real-world messy data.
        1. Drops ID columns (heuristic: all unique values).
        2. Encodes strings.
        3. Fills NaNs.
        """
        # Drop the target from features
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Detect problem type (Regression vs Classification)
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
            self.problem_type = "regression"
        else:
            self.problem_type = "classification"
            # Encode target if it's a string (e.g., "Yes"/"No")
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

        # Handle Categorical Features (Simple Label Encoding for speed)
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            # Fill NaNs with 0 (Rough fix for MVP)
            X[col] = X[col].fillna(0)
            
        return X, y

    def train_forensic_model(self):
        print(f"⚙️ Training Forensic {self.problem_type.capitalize()} Model...")
        
        X, y = self.preprocess_data()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_test = X_test
        
        # Select Model based on problem type
        if self.problem_type == "classification":
            self.model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            
        self.model.fit(X_train, y_train)
        
        # Score
        acc = self.model.score(X_test, y_test)
        return acc

    def analyze_with_shap(self):
        print("🕵️ Calculating SHAP Values...")
        
        # TreeExplainer is fast for Random Forests
        explainer = shap.TreeExplainer(self.model)
        
        # Sample to keep it fast
        if len(self.X_test) > 200:
            sample_X = self.X_test.sample(200, random_state=42)
        else:
            sample_X = self.X_test
            
        # calculate
        raw_shap_values = explainer.shap_values(sample_X)
        self.X_test_sample = sample_X 

        # --- THE FIX: HANDLING CLASS OUTPUTS ---
        
        # Scenario A: It returns a list of arrays (e.g. [Class0_Array, Class1_Array])
        if isinstance(raw_shap_values, list):
            print(f"Debug: SHAP returned a list of length {len(raw_shap_values)}")
            # For binary, usually index 1 is the 'Positive' class. 
            # If only 1 item in list (Regression), take index 0.
            if len(raw_shap_values) > 1:
                self.shap_values = raw_shap_values[1]
            else:
                self.shap_values = raw_shap_values[0]
                
        # Scenario B: It returns a numpy array
        else:
            self.shap_values = np.array(raw_shap_values)
            print(f"Debug: SHAP returned array of shape {self.shap_values.shape}")
            
            # If shape is (Samples, Features, Classes) -> e.g. (200, 31, 2)
            if len(self.shap_values.shape) == 3:
                # Take the slice for Class 1
                self.shap_values = self.shap_values[:, :, 1]
                
        # Final Check
        print(f"Debug: Final SHAP Shape: {self.shap_values.shape}")