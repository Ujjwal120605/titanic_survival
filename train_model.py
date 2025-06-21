# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import seaborn as sns

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Select features and drop missing values
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'fare']].dropna()

# Encode 'sex' column
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'titanic_model.pkl')
print("✅ Model saved as titanic_model.pkl")

