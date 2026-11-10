
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible synthetic sensor data
rng = np.random.default_rng(42)
n_samples = 5000

# Simulate two operating conditions
failure = rng.choice(
    [0, 1],
    size=n_samples,
    p=[0.80, 0.20]
)

# Sensor readings for normal operation and failure conditions
temperature = np.where(
    failure == 0,
    rng.normal(65, 7, n_samples),
    rng.normal(90, 9, n_samples)
)

vibration = np.where(
    failure == 0,
    rng.normal(2.0, 0.7, n_samples),
    rng.normal(5.0, 1.2, n_samples)
)

pressure = np.where(
    failure == 0,
    rng.normal(5.0, 0.5, n_samples),
    rng.normal(3.8, 0.8, n_samples)
)

rpm = np.where(
    failure == 0,
    rng.normal(1500, 100, n_samples),
    rng.normal(1250, 180, n_samples)
)

# Keep sensor values within plausible simulated limits
temperature = np.clip(temperature, 35, 125)
vibration = np.clip(vibration, 0.1, 9)
pressure = np.clip(pressure, 1, 8)
rpm = np.clip(rpm, 500, 2000)

df = pd.DataFrame({
    "temperature": temperature.round(2),
    "vibration": vibration.round(2),
    "pressure": pressure.round(2),
    "rpm": rpm.round(0).astype(int),
    "failure": failure
})

df["status"] = df["failure"].map({
    0: "Normal",
    1: "Failure"
})

# Save dataset in the project's data folder
output_path = Path("data") / "machine_sensor_data.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("Dataset generated successfully!")
print(f"Saved to: {output_path}")
print(f"Total readings: {len(df)}")
print("\nClass distribution:")
print(df["status"].value_counts())
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
