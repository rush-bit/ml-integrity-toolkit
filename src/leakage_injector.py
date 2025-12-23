import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

class LeakageInjector:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(self.seed)

    def load_data(self):
        """
        Loads a clean standard dataset (Breast Cancer)
        """
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df

    def inject_target_leakage(self, df, noise_level=0.1):
        """
        Creates a 'God Feature' that is basically the target with slight noise.
        This simulates a feature that accidentally contains the answer.
        """
        print(f"Injecting Target Leakage (Noise Level: {noise_level})...")
        
        # Create a feature that IS the target + random noise
        # This is the most common type of fatal leakage
        noise = np.random.normal(0, noise_level, size=len(df))
        df['leaky_feature_1'] = df['target'] + noise
        
        return df

    def inject_id_leakage(self, df):
        """
        Creates a proxy ID feature that correlates with the target by accident.
        Simulates: 'Patient_ID' sorted by severity of disease.
        """
        print("Injecting ID/Proxy Leakage...")
        
        # Sort by target so index correlates with target, then add noise
        df = df.sort_values(by='target').reset_index(drop=True)
        df['leaky_id'] = df.index + np.random.randint(0, 10, size=len(df))
        
        # Shuffle back so it's not obvious
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        return df

if __name__ == "__main__":
    injector = LeakageInjector()
    
    # 1. Load Clean Data
    df = injector.load_data()
    print(f"Loaded Clean Data: {df.shape}")
    df.to_csv("data/clean_data.csv", index=False)
    
    # 2. Inject Leakage
    df_leaky = injector.inject_target_leakage(df)
    df_leaky = injector.inject_id_leakage(df_leaky)
    
    # 3. Save Corrupted Data
    print(f"Saving Corrupted Data: {df_leaky.shape}")
    df_leaky.to_csv("data/leaky_data.csv", index=False)
    print("Done. Check your 'data/' folder.")