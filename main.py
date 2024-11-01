import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

should_be_simple = True  # make a really simple data set for demo purposes
force_fresh_data = True  # ignore existing data (csv file) and generate new data

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

if os.path.exists(csv_filename) and not force_fresh_data:
    # Load the dataset from the CSV file
    c.print(f"[yellow]Loading data from {csv_filename}...[/yellow]")
    data = pd.read_csv(csv_filename)
    c.print(f"Data loaded from {csv_filename}.")
else:
    c.print("[yellow]Generating synthetic data...[/yellow]")
    # Number of samples
    n_samples = 10

    if should_be_simple:
        # Simple synthetic dataset
        data = pd.DataFrame(
            {
                "DiagnosisCode": np.random.choice(["X"], n_samples),
                "ProcedureCode": np.random.choice(["A"], n_samples),
                "AvgLengthOfStay": np.random.randint(5, n_samples),
                "PatientAge": np.random.randint(18, 100, n_samples),
                "PatientGender": np.random.choice(["Male", "Female"], n_samples),
                "DaysToProcess": np.random.randint(1, 30, n_samples),
                "ClaimAmount": np.round(np.random.uniform(100, n_samples), 2),
            }
        )
    else:
        # Generate synthetic features
        data = pd.DataFrame(
            {
                "DiagnosisCode": np.random.choice(["X", "Y", "Z"], n_samples),
                "ProcedureCode": np.random.choice(["A", "B", "C", "D"], n_samples),
                "AvgLengthOfStay": np.random.randint(5, n_samples),
                "PatientAge": np.random.randint(18, 100, n_samples),
                "PatientGender": np.random.choice(["Male", "Female"], n_samples),
                "DaysToProcess": np.random.randint(1, 30, n_samples),
                "ClaimAmount": np.round(np.random.uniform(100, 10000, n_samples), 2),
            }
        )

    # Simulate the delta in payment (target variable)
    def simulate_delta(row):
        delta = 0

        if should_be_simple:
            if row["PatientGender"] == "Male":
                delta -= np.random.uniform(50, 200)
            # Introduce randomness
            # delta += np.random.normal(0, 50)
            return max(delta, 0)  # Delta should not be negative
        else:

            if row["ProcedureCode"] == "A":
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
categorical_features = ["ProcedureCode", "PatientGender", "DiagnosisCode"]
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
plt.savefig("plots/actual_vs_predicted.png")
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
