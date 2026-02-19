import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, classification_report

def verify_clinical_performance():
    print("🧪 Testing MediDiagnose AI on Lucknow Clinical Data...")
    
    # 1. Load the AI Brain
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        features = pickle.load(open('features.pkl', 'rb'))
        print(f"🧠 Model loaded. Expecting {len(features)} features.")
    except FileNotFoundError:
        print("❌ Error: 'model.pkl' or 'features.pkl' not found.")
        return

    # 2. Load the Prepared Test Data
    try:
        test_df = pd.read_csv('test_ready_data.csv')
    except FileNotFoundError:
        print("❌ Error: 'test_ready_data.csv' not found. Run preprocessor first.")
        return
    
    # 3. FEATURE ALIGNMENT LAYER
    # Fills missing columns (LY%, MO%, is_smoker, etc.) with 0 
    for col in features:
        if col not in test_df.columns:
            test_df[col] = 0
            
    # Ensure WBC is mapped from TLC if the model needs it
    if 'WBC' in features and 'WBC' not in test_df.columns:
        if 'TLC' in test_df.columns:
            test_df['WBC'] = test_df['TLC']

    X_test = test_df[features]
    y_true = test_df['Target']
    
    # 4. Run AI Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    
    # 5. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    
    try:
        loss = log_loss(y_true, y_probs, labels=[0, 1])
        print(f"📉 Validation Log Loss: {loss:.4f}")
    except ValueError:
        print("📉 Validation Log Loss: N/A")

    print(f"✅ Final Accuracy on Clinical Data: {acc * 100:.2f}%")
    print("-" * 40)
    
    # 6. Confusion Matrix Visualization
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PuRd', 
                xticklabels=['Healthy (0)', 'Anemic (1)'], 
                yticklabels=['Healthy (0)', 'Anemic (1)'])
    
    plt.title('Clinical Validation: MediDiagnose AI')
    plt.xlabel('AI Predicted')
    plt.ylabel('Actual Truth')
    
    plt.savefig('lucknow_performance_matrix.png')
    print("🖼️ Confusion Matrix saved as 'lucknow_performance_matrix.png'")
    
    # 7. Detailed Stats
    print("\nDetailed Clinical Classification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1], 
                                target_names=['Healthy', 'Anemic'], 
                                zero_division=0))

if __name__ == "__main__":
    verify_clinical_performance()