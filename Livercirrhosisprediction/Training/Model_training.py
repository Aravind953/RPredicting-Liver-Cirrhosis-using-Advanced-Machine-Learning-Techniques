import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv('../Data/liver_dataset.csv')

# Convert Gender to numeric if necessary
if df['Gender'].dtype == 'object':
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Split features and label
X = df.drop("Target", axis=1)
y = df["Target"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
pickle.dump(model, open("../Flask/rf_acc_68.pkl", "wb"))
pickle.dump(scaler, open("../Flask/normalizer.pkl", "wb"))

print("✅ Model and scaler saved successfully.")