"""
Script to retrain the model with current sklearn version.
This fixes the pickle compatibility issue.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Read data
data_path = os.path.join("artifacts_reference", "data.csv")
df = pd.read_csv(data_path)

print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Define features and target
target_column = "math_score"
numerical_columns = ["writing_score", "reading_score"]
categorical_columns = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]

# Split features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Create preprocessing pipelines
num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
        ("scaler", StandardScaler(with_mean=False))
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num_pipeline", num_pipeline, numerical_columns),
        ("cat_pipelines", cat_pipeline, categorical_columns)
    ]
)

# Fit the preprocessor and transform data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print(f"Transformed training shape: {X_train_transformed.shape}")

# Train a simple RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_transformed, y_train)

# Evaluate
train_score = model.score(X_train_transformed, y_train)
test_score = model.score(X_test_transformed, y_test)

print(f"Training R² score: {train_score:.4f}")
print(f"Test R² score: {test_score:.4f}")

# Save the preprocessor and model
output_dir = "artifacts_reference"

preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
model_path = os.path.join(output_dir, "model.pkl")

with open(preprocessor_path, "wb") as f:
    pickle.dump(preprocessor, f)
print(f"Saved preprocessor to {preprocessor_path}")

with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Saved model to {model_path}")

print("\n✅ Model retraining complete! The app should work now.")
