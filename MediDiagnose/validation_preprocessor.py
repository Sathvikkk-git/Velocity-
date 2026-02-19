import pandas as pd
import numpy as np

def prepare_lucknow_data(input_file):
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # 1. Strip spaces from column names
    df.columns = df.columns.str.strip()
    print("Fixed Columns:", df.columns.tolist())

    # 2. DATA SANITIZER: Force numeric columns to be numbers
    cols_to_fix = ['RBC', 'MCV', 'HGB', 'PCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT /mm3']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. MEDICAL MAPPING
    mapping = {
        'HGB': 'HGB',
        'Sex': 'Gender',
        'TLC': 'WBC',
        'PCV': 'HCT',
        'PLT /mm3': 'PLT'
    }
    df = df.rename(columns=mapping)

    # --- NEW: CLINICAL FEATURE RE-ENGINEERING ---
    # We estimate these indices so the model doesn't see "0" values.
    
    # Hematocrit (HCT) estimation using the "Rule of Three"
    if 'HCT' not in df.columns:
        df['HCT'] = df['HGB'] * 3
    
    # MCH: (HGB / RBC) * 10
    df['MCH'] = (df['HGB'] / (df['RBC'] + 1e-6)) * 10
    
    # MCHC: (HGB / HCT) * 100
    df['MCHC'] = (df['HGB'] / (df['HCT'] + 1e-6)) * 100
    
    # Ensure all required columns for the 23-feature model exist
    extra_features = ['is_smoker', 'is_athlete', 'LY%', 'MO%', 'NE%', 'EO%', 'BA%', 'MPV', 'WBC', 'RDW']
    for feat in extra_features:
        if feat not in df.columns:
            df[feat] = 0.0

    # 4. FEATURE ENGINEERING
    df['Mentzer_Index'] = df['MCV'] / (df['RBC'] + 1e-6)
    
    # 5. DEFINE THE TARGET (Anemia Labeling)
    if 'Gender' in df.columns:
        df['Gender_Clean'] = df['Gender'].astype(str).str.strip().str.upper()
        df['Target'] = np.where(
            ((df['Gender_Clean'].isin(['M', '0', 'MALE'])) & (df['HGB'] < 13)) | 
            ((df['Gender_Clean'].isin(['F', '1', 'FEMALE'])) & (df['HGB'] < 12)), 
            1, 0
        )
    else:
        df['Target'] = np.where(df['HGB'] < 12.5, 1, 0)

    # 6. FINAL CLEANING
    df = df.fillna(df.median(numeric_only=True))
    
    print(f"✅ Processed {len(df)} patient records.")
    df.to_csv('test_ready_data.csv', index=False)
    print("🚀 'test_ready_data.csv' is ready for accuracy testing!")

if __name__ == "__main__":
    prepare_lucknow_data('lucknow_dataset.csv')