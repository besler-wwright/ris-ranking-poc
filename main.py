import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Initialize console and environment
c = Console()
os.system("cls" if os.name == "nt" else "clear")
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Set data directory
data_dir = "data"

# Set random seed for reproducibility
np.random.seed(42)

# Filename for the synthesized dataset
csv_filename = "data/synthesized_medical_claims.csv"

if os.path.exists(csv_filename):
    # Load the dataset from the CSV file
    data = pd.read_csv(csv_filename)
    print(f"Data loaded from {csv_filename}.")
else:
    # Number of samples
    n_samples = 200

    # Generate synthetic features
    data = pd.DataFrame(
        {
            "ClaimAmount": np.random.uniform(100, 10000, n_samples),
            "ProcedureCode": np.random.choice(["A", "B", "C", "D"], n_samples),
            "ProviderType": np.random.choice(["General", "Specialist"], n_samples),
            "PatientAge": np.random.randint(0, 100, n_samples),
            "PatientGender": np.random.choice(["Male", "Female"], n_samples),
            "DiagnosisCode": np.random.choice(["X", "Y", "Z"], n_samples),
            "DaysToProcess": np.random.randint(1, 30, n_samples),
        }
    )

    # Simulate the delta in payment (target variable)
    def simulate_delta(row):
        delta = 0
        if row["ProcedureCode"] == "A" and row["ProviderType"] == "Specialist":
            delta += np.random.uniform(100, 500)
        if row["PatientAge"] > 65:
            delta += np.random.uniform(50, 300)
        if row["DiagnosisCode"] == "Z":
            delta -= np.random.uniform(50, 200)
        # Introduce randomness
        delta += np.random.normal(0, 50)
        return max(delta, 0)  # Delta should not be negative

    data["DeltaPayment"] = data.apply(simulate_delta, axis=1)

    # Save the synthesized dataset to CSV
    data.to_csv(csv_filename, index=False)
    print(f"Synthesized data saved to {csv_filename}.")

# One-hot encode categorical variables
categorical_features = ["ProcedureCode", "ProviderType", "PatientGender", "DiagnosisCode"]
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Features and target
X = data_encoded.drop("DeltaPayment", axis=1)
y = data_encoded["DeltaPayment"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on test set: {mae:.2f}")

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual DeltaPayment")
plt.ylabel("Predicted DeltaPayment")
plt.title("Actual vs. Predicted DeltaPayment")
plt.savefig('plots/actual_vs_predicted.png')
plt.close()
print("Plot saved as 'plots/actual_vs_predicted.png'")

# Rank new claims
X_new = X_test.copy()
delta_predictions = model.predict(X_new)
results = X_new.copy()
results["PredictedDeltaPayment"] = delta_predictions
results_sorted = results.sort_values(by="PredictedDeltaPayment", ascending=False)

print("Top 5 claims likely needing rework:")
print(results_sorted.head(5)[["PredictedDeltaPayment"]])
