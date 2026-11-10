
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
DATA_PATH = Path("data/machine_sensor_data.csv")
MODEL_PATH = Path("models/machine_failure_model.joblib")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Select sensor features and prediction target
features = ["temperature", "vibration", "pressure", "rpm"]
X = df[features]
y = df["failure"]

# Split data before training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate on unseen test data
y_pred = model.predict(X_test)

print("=" * 50)
print("EDGEGUARD MODEL EVALUATION")
print("=" * 50)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Accuracy:         {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Failure"],
    zero_division=0
))

print("Confusion matrix:")
print("Rows = actual; columns = predicted")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Save model and feature schema
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": features,
        "target": "failure",
        "class_names": {0: "Normal", 1: "Failure"}
    },
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")
print("Training completed successfully!")
