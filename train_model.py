import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Load and fix CSV
df = pd.read_csv("dataset/A_to_Z_dataset.csv")

# Extract features and labels
X_raw = df.drop('label', axis=1).values
y = df['label'].values

# Clean: convert all values to float safely
X = []
for row in X_raw:
    fixed_row = []
    for val in row:
        try:
            fixed_row.append(float(val))
        except:
            fixed_row.append(0.0)
    X.append(fixed_row)
X = np.array(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "model/label_encoder.pkl")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "model/scaler.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {round(acc * 100, 2)}%")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/gesture_model.pkl")
print("📁 Saved model to: model/gesture_model.pkl")
