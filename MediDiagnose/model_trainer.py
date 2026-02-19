import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_medidiagnose():
    print("🧠 Training the AI Brain...")
    
    # 1. Load the processed data
    df = pd.read_csv('processed_cbc_data.csv')
    
    # 2. Select Features (The markers the AI uses to decide)
    # We drop 'Target' because that's the answer, and 'Risk_Score' 
    # because it's a derived label, not a raw input.
    X = df.drop(columns=['Target', 'Risk_Score', 'Is_Emergency'], errors='ignore')
    y = df['Target']
    
    # 3. Split into Train (80%) and Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize Random Forest
    # n_estimators=100 means 100 decision trees voting together
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # 5. Train the model
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✅ Training Complete! Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, predictions))
    
    # 7. SAVE THE MODEL (Crucial for the App)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Also save the feature names to ensure the App uses them in the right order
    with open('features.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
        
    print("💾 Model and Features saved as 'model.pkl' and 'features.pkl'")

if __name__ == "__main__":
    train_medidiagnose()