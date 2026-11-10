
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

MODEL_PATH = Path("models/machine_failure_model.joblib")
LOG_PATH = Path("data/prediction_history.csv")

# Load the already-trained model
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

rng = np.random.default_rng()

def generate_sensor_reading():
    """Simulate one new machine sensor reading."""

    # Occasionally simulate abnormal machine conditions.
    abnormal = rng.random() < 0.25

    if abnormal:
        temperature = rng.normal(90, 8)
        vibration = rng.normal(5.0, 1.0)
        pressure = rng.normal(3.8, 0.7)
        rpm = rng.normal(1250, 150)
    else:
        temperature = rng.normal(65, 7)
        vibration = rng.normal(2.0, 0.6)
        pressure = rng.normal(5.0, 0.4)
        rpm = rng.normal(1500, 90)

    return {
        "temperature": round(float(np.clip(temperature, 35, 125)), 2),
        "vibration": round(float(np.clip(vibration, 0.1, 9)), 2),
        "pressure": round(float(np.clip(pressure, 1, 8)), 2),
        "rpm": int(np.clip(round(rpm), 500, 2000))
    }

print("EDGEGUARD - REAL-TIME MACHINE MONITOR")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        reading = generate_sensor_reading()

        # Maintain the exact feature order used in training.
        X_new = pd.DataFrame([reading], columns=features)

        prediction = int(model.predict(X_new)[0])
        probabilities = model.predict_proba(X_new)[0]
        failure_index = list(model.classes_).index(1)
        failure_probability = float(probabilities[failure_index])

        status = "FAILURE ALERT" if prediction == 1 else "NORMAL"

        timestamp = datetime.now().isoformat(timespec="seconds")

        print("-" * 55)
        print(f"Time:        {timestamp}")
        print(f"Temperature: {reading['temperature']} °C")
        print(f"Vibration:   {reading['vibration']} mm/s")
        print(f"Pressure:    {reading['pressure']} bar")
        print(f"RPM:         {reading['rpm']}")
        print(f"Prediction:  {status}")
        print(f"Model failure probability: {failure_probability:.1%}")

        record = {
            "timestamp": timestamp,
            **reading,
            "predicted_failure": prediction,
            "failure_probability": round(failure_probability, 4)
        }

        # Append each reading to the history CSV.
        pd.DataFrame([record]).to_csv(
            LOG_PATH,
            mode="a",
            header=not LOG_PATH.exists(),
            index=False
        )

        time.sleep(2)

except KeyboardInterrupt:
    print("\nSensor simulation stopped.")
