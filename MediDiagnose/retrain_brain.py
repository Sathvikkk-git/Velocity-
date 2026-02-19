import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load your original training data
# (Ensure this file is in your folder)
df = pd.read_csv('processed_cbc_data.csv') 

# 2. Select the "Clinical Core Five"
# These are universal across NHANES and Lucknow datasets
features = ['HGB', 'RBC', 'MCV', 'MCH', 'RDW']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the "Lean" Random Forest
# We use a balanced depth to prevent overfitting
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 4. Quick Verify
y_pred = model.predict(X_test)
print(f"✅ Retraining Complete. Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 5. Save the new Brain
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(features, open('features.pkl', 'wb'))
print("💾 'model.pkl' and 'features.pkl' updated successfully.")