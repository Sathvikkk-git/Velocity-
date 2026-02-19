import pandas as pd
import numpy as np

def prepare_medical_data(filepath):
    df = pd.read_csv("cbc_dataframe.csv")
    
    # 1. DATA CLEANING
    garbage_cols = ['Patient_ID', 'Index', 'Unnamed: 0', 'id', 'ID']
    df = df.drop(columns=[col for col in garbage_cols if col in df.columns])
    
    # Robust Median Filling
    df = df.fillna(df.median())

    # 2. FEATURE ENGINEERING (Mentzer Index)
    # Adding 1e-6 to avoid DivisionByZero errors
    df['Mentzer_Index'] = df['MCV'] / (df['RBC'] + 1e-6)

    # 3. SMOKER’S SIGNATURE (Injection)
    df['is_smoker'] = 0
    smoker_indices = df.sample(frac=0.10, random_state=42).index
    df.loc[smoker_indices, 'is_smoker'] = 1
    df.loc[smoker_indices, 'WBC'] *= 1.3  
    df.loc[smoker_indices, 'HGB'] *= 1.05 

    # 4. ATHLETE’S SIGNATURE (The "Pseudo-Anemia" logic)
    # High physical activity can increase plasma volume
    df['is_athlete'] = 0
    athlete_indices = df.sample(frac=0.05, random_state=7).index
    df.loc[athlete_indices, 'is_athlete'] = 1
    df.loc[athlete_indices, 'HGB'] *= 0.95 # Looks low, but is actually healthy

    # 5. REFINED TARGET & RISK (WHO Standards)
    def calculate_target_and_risk(row):
        # Determine threshold based on Gender (0=Male, 1=Female)
        # If no gender col, default to 12.5
        threshold = 13.0 if row.get('Gender', 0) == 0 else 12.0
        
        # Target Label
        is_anemic = 1 if row['HGB'] < threshold else 0
        
        # Risk Score (For Healthy People)
        risk = -1 # Default for already anemic
        if is_anemic == 0:
            if row['RDW'] < 14.0: risk = 0 # Optimal
            elif 14.0 <= row['RDW'] <= 15.5: risk = 1 # Medium
            else: risk = 2 # High
            
        return pd.Series([is_anemic, risk])

    df[['Target', 'Risk_Score']] = df.apply(calculate_target_and_risk, axis=1)

    # 6. EMERGENCY TRIAGE
    df['Is_Emergency'] = (df['HGB'] < 7.0).astype(int)

    return df

if __name__ == "__main__":
    processed_df = prepare_medical_data('C:\\Users\\Sathvik\\MediDiagnose\\cbc_dataframe.csv')
    processed_df.to_csv('processed_cbc_data.csv', index=False)
    print("✅ Research-Grade Dataset Created: 'processed_cbc_data.csv'")