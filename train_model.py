# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("/Users/anshusingh/Desktop/Ecommerce_fixed-1.csv")

features = [
    'view_count', 'cart_additions', 'time_spent',
    'price', 'discount', 'rating',
    'past_purchases', 'avg_spent'
]

X = df[features]
y = df['purchased']

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved!")