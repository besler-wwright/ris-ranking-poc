import csv
import os

from rich.console import Console

# Initialize console and environment
c = Console()
os.system("cls" if os.name == "nt" else "clear")
c.print("[bold green]RIS Ranking Proof of Concept[/bold green]")

import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_claim_scoring_model(claims_data, features):
    """
    Train a model to score claims based on rework probability and payment impact.

    Parameters:
    claims_data (pd.DataFrame): DataFrame containing claims data
    features (list): List of feature columns to use for prediction

    Returns:
    tuple: (trained_model, scaler, feature_importance)
    """
    # Prepare the data
    X = claims_data[features]
    y = claims_data["needs_rework"]  # Binary target variable

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Get feature importance
    feature_importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)

    return model, scaler, feature_importance


def score_claims(model, scaler, new_claims, features):
    """
    Score new claims based on rework probability and potential payment impact.

    Parameters:
    model: Trained model
    scaler: Fitted StandardScaler
    new_claims (pd.DataFrame): New claims to score
    features (list): List of feature columns used in training

    Returns:
    pd.DataFrame: Original claims data with priority scores
    """
    # Scale the features
    X_new_scaled = scaler.transform(new_claims[features])

    # Get probability of rework needed
    rework_prob = model.predict_proba(X_new_scaled)[:, 1]

    # Calculate expected payment impact
    # Using historical average payment difference for reworked claims
    avg_payment_diff = new_claims["payment_difference"].mean()
    expected_impact = rework_prob * avg_payment_diff

    # Calculate priority score (combining probability and impact)
    # Normalize both components to 0-1 scale
    normalized_prob = rework_prob / rework_prob.max()
    normalized_impact = expected_impact / expected_impact.max()

    # Combine scores (equal weight to probability and impact)
    priority_score = (normalized_prob + normalized_impact) / 2

    # Add scores to the claims data
    scored_claims = new_claims.copy()
    scored_claims["rework_probability"] = rework_prob
    scored_claims["expected_impact"] = expected_impact
    scored_claims["priority_score"] = priority_score

    return scored_claims.sort_values("priority_score", ascending=False)


def generate_synthetic_claims(
    df_name_prefix="MyData",
    num_claims=1000,
    seed=42,
    num_of_providers=1,
    num_of_diagnosis_codes=1,
    num_of_procedure_codes=1,
    specialties=["Internal Med", "Cardiology", "Orthopedics", "Neurology", "General Surgery"],
    should_display_stats=False,
):
    """
    Generate synthetic medical claims data with realistic patterns.
    """
    np.random.seed(seed)

    # Create lists for categorical variables
    provider_ids = [f"PRV{str(i).zfill(4)}" for i in range(1, num_of_providers + 1)]  # 50 providers
    diagnosis_codes = [f"ICD{str(i).zfill(3)}" for i in range(1, num_of_diagnosis_codes + 1)]  # 30 diagnosis codes
    procedure_codes = [f"CPT{str(i).zfill(4)}" for i in range(1, num_of_procedure_codes + 1)]  # 40 procedure codes

    # Generate diagnosis-specific average LOS
    diagnosis_los = {}
    for code in diagnosis_codes:
        # Different diagnoses have different typical LOS
        if code.startswith("ICD00"):  # Complex conditions
            diagnosis_los[code] = np.round(np.random.uniform(5, 10), 1)
        elif code.startswith("ICD01"):  # Moderate conditions
            diagnosis_los[code] = np.round(np.random.uniform(3, 6), 1)
        else:  # Less complex conditions
            diagnosis_los[code] = np.round(np.random.uniform(1, 4), 1)

    # Generate procedure-specific charges
    procedure_charges = {}
    for code in procedure_codes:
        # Different procedures have different costs
        if code.startswith("CPT00"):  # Complex procedures
            procedure_charges[code] = round(np.random.uniform(4000, 5000), 2)
        elif code.startswith("CPT01"):  # Medium procedures
            procedure_charges[code] = round(np.random.uniform(2500, 4000), 2)
        else:  # Simpler procedures
            procedure_charges[code] = round(np.random.uniform(1000, 2500), 2)

    # Generate base data
    # Generate procedure codes first
    procedure_code_list = np.random.choice(procedure_codes, num_claims)

    data = {
        "claim_id": [f"CLM{str(i).zfill(6)}" for i in range(1, num_claims + 1)],
        "date_submitted": [(datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime("%Y-%m-%d") for _ in range(num_claims)],
        "provider_id": np.random.choice(provider_ids, num_claims),
        "provider_specialty": np.random.choice(specialties, num_claims),
        "diagnosis_code": np.random.choice(diagnosis_codes, num_claims),
        "procedure_code": procedure_code_list,
        "claim_charges": [procedure_charges[code] for code in procedure_code_list],
    }

    # Add derived features that might influence rework probability
    df = pd.DataFrame(data)

    # Add LOS features
    df["avg_los"] = df["diagnosis_code"].map(diagnosis_los)

    # Generate actual LOS with some variation around the expected LOS
    df["actual_los"] = df.apply(lambda row: max(1, np.random.normal(row["avg_los"], row["avg_los"] * 0.2)), axis=1).round(1)  # 20% standard deviation

    # Calculate LOS difference (actual - expected)
    df["los_difference"] = (df["actual_los"] - df["avg_los"]).round(1)

    # Generate 'needs_rework' based on various factors including LOS
    rework_probabilities = (
        # Base probability
        np.random.random(num_claims) * 0.02
        +
        # Higher amounts more likely to need rework
        (df["claim_charges"] > 1000).astype(float) * 0.1
        +
        # Certain procedures more likely to need rework
        (df["procedure_code"].isin(["CPT0001", "CPT0002", "CPT0003"])).astype(float) * 0.15
        +
        # Certain providers more likely to need rework
        (df["provider_id"].isin(["PRV0001", "PRV0002"])).astype(float) * 0.2
        +
        # LOS difference increases rework probability
        (abs(df["los_difference"]) > 2).astype(float) * 0.2
    )

    df["needs_rework"] = (rework_probabilities > 0.5).astype(int)

    # Generate payment difference for reworked claims
    df["payment_difference"] = 0.0
    rework_mask = df["needs_rework"] == 1

    # Payment difference is related to original claim amount and LOS difference
    df.loc[rework_mask, "payment_difference"] = df.loc[rework_mask, "claim_charges"] * (
        np.random.uniform(-0.3, 0.3, size=rework_mask.sum()) + df.loc[rework_mask, "los_difference"] * 0.05
    )  # LOS difference affects payment

    # Round monetary values to 2 decimal places
    df["claim_charges"] = df["claim_charges"].round(2)
    df["payment_difference"] = df["payment_difference"].round(2)

    # Convert date_submitted to datetime
    df["date_submitted"] = pd.to_datetime(df["date_submitted"])

    # Sort by date
    df = df.sort_values("date_submitted")

    # Reorder columns
    column_order = [
        "claim_id",
        "date_submitted",
        "diagnosis_code",
        "procedure_code",
        "claim_charges",
        "avg_los",
        "actual_los",
        "los_difference",
        "provider_id",
        "provider_specialty",
        "needs_rework",
        "payment_difference",
    ]
    df = df[column_order]

    # Save to CSV
    csv_filename = f"data/{df_name_prefix}_synthesized_medical_claims.csv"
    df.to_csv(csv_filename, index=False)
    c.print(f"[bold green]{df_name_prefix} Saved to {csv_filename}.[/bold green]")

    if should_display_stats:
        display_stats_for_df(df)

    return df


def prepare_features(df):
    """
    Prepare features for the model, including encoding and feature engineering.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()

    # Encode categorical variables
    le = LabelEncoder()
    data["provider_id_encoded"] = le.fit_transform(data["provider_id"])
    data["procedure_code_encoded"] = le.fit_transform(data["procedure_code"])
    data["diagnosis_code_encoded"] = le.fit_transform(data["diagnosis_code"])
    data["provider_specialty_encoded"] = le.fit_transform(data["provider_specialty"])

    # Create feature list for model
    features = ["claim_charges", "provider_id_encoded", "procedure_code_encoded", "diagnosis_code_encoded", "provider_specialty_encoded", "avg_los", "actual_los", "los_difference"]

    return data, features


def train_and_evaluate_random_forest_model(data, features):
    """
    Train the model and evaluate its performance.
    """
    X = data[features]
    y = data["needs_rework"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Get feature importance
    feature_importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)

    # Get predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    return model, scaler, feature_importance, X_test, y_test, y_pred_proba


def calculate_priority_scores(data, model, scaler, features):
    """
    Calculate priority scores combining rework probability and potential impact.
    """
    X = data[features]
    X_scaled = scaler.transform(X)

    # Get rework probabilities
    rework_prob = model.predict_proba(X_scaled)[:, 1]

    # Calculate expected impact based on both payment difference and LOS difference
    avg_payment_impact = abs(data["payment_difference"]).mean()
    avg_los_impact = abs(data["los_difference"]).mean()

    # Normalize payment and LOS impacts to 0-1 scale
    normalized_payment_impact = abs(data["payment_difference"]) / abs(data["payment_difference"]).max()
    normalized_los_impact = abs(data["los_difference"]) / abs(data["los_difference"]).max()

    # Combined impact score (weighted average of payment and LOS impact)
    impact_score = normalized_payment_impact * 0.7 + normalized_los_impact * 0.3

    # Calculate final priority score
    priority_score = rework_prob * 0.6 + impact_score * 0.4

    return rework_prob, impact_score, priority_score


def display_stats_for_df(claims_df):
    """
    Display various statistics for a given DataFrame containing claims data.

    Parameters:
    claims_df (pandas.DataFrame): A DataFrame containing claims data with the following columns:
        - procedure_code: The code for the medical procedure.
        - avg_los: The average length of stay.
        - actual_los: The actual length of stay.
        - los_difference: The difference between the average and actual length of stay.
        - needs_rework: A binary indicator of whether rework is needed.

    The function prints:
    - The first few rows of the dataset.
    - Summary statistics for the dataset.
    - LOS (Length of Stay) statistics grouped by procedure type.
    - Correlation between LOS difference and the need for rework.
    """

    c = Console()
    c.print("\n[black]-----First few rows of the dataset:[/black]--------------------------------------")
    c.print(claims_df.head())

    # c.print("\nSummary statistics:")
    # c.print(claims_df.describe())

    c.print("\n[black]-----LOS statistics by procedure type:[/black]--------------------------------")
    c.print(claims_df.groupby("procedure_code")[["avg_los", "actual_los", "los_difference"]].mean())

    c.print("\n[black]-----Correlation between LOS difference and rework:[/black]-------------------")
    c.print(claims_df["los_difference"].abs().corr(claims_df["needs_rework"]))


def score_data(df_name_prefix, claims_df, rework_prob, impact_score, priority_score):
    scored_claims = claims_df.copy()
    scored_claims["rework_probability"] = rework_prob
    scored_claims["impact_score"] = impact_score
    scored_claims["priority_score"] = priority_score

    # Sort by priority score
    scored_claims_sorted = scored_claims.sort_values("priority_score", ascending=False)

    # Reorder columns before saving
    column_order = [
        "claim_id",
        "date_submitted",
        "diagnosis_code",
        "procedure_code",
        "claim_charges",
        "avg_los",
        "actual_los",
        "los_difference",
        "provider_id",
        "provider_specialty",
        "needs_rework",
        "payment_difference",
    ]
    scored_claims_sorted = scored_claims_sorted[column_order + ["rework_probability", "impact_score", "priority_score"]]

    # Save scored claims to CSV
    csv_filename = f"data/{df_name_prefix}_scored_medical_claims.csv"
    scored_claims_sorted.to_csv(csv_filename, index=False)
    c.print(f"[green]{df_name_prefix}Scored Data Saved to {csv_filename}.[/green]")
    return scored_claims_sorted


def run_random_forest_model(claims_df, should_display_stats=False):
    c = Console()

    # prepare the data
    prepared_data, features = prepare_features(claims_df)

    # Train and evaluate the model
    model, scaler, feature_importance, X_test, y_test, y_pred_proba = train_and_evaluate_random_forest_model(prepared_data, features)

    # Calculate priority scores for all claims
    rework_prob, impact_score, priority_score = calculate_priority_scores(prepared_data, model, scaler, features)

    predictions = (y_pred_proba > 0.5).astype(int)

    # Print prediction distributions
    if should_display_stats:
        c.print("\nClass distribution in predictions:", np.unique(predictions, return_counts=True))
        c.print("Class distribution in test set:", np.unique(y_test, return_counts=True))

        c.print("\nModel Performance:")
        c.print(classification_report(y_test, predictions, zero_division=0))

    return feature_importance, rework_prob, impact_score, priority_score


def run_random_forest_and_score_data(df_name_prefix, claims_df):
    c = Console()
    # Model the data
    feature_importance, rework_prob, impact_score, priority_score = run_random_forest_model(claims_df)

    # Add scores to the original dataframe
    scored_claims_sorted = score_data(df_name_prefix, claims_df, rework_prob, impact_score, priority_score)

    # Print summary statistics and top priority claims
    c.print("\n[bold green]-----Feature Importance:[/bold green]--------------------------------")
    c.print(feature_importance)

    c.print("\n[black]-----Top 10 Priority Claims:[/black]--------------------------------")
    c.print(
        scored_claims_sorted[["claim_id", "diagnosis_code", "claim_charges", "rework_probability", "impact_score", "priority_score", "los_difference", "payment_difference"]].head(
            10
        )
    )


# Start ---------------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# Generate the dataset
df_name_prefix = "SIMPLE"
claims_df = generate_synthetic_claims(
    df_name_prefix,
    num_claims=100,
    seed=42,
    num_of_providers=1,
    num_of_diagnosis_codes=1,
    num_of_procedure_codes=1,
    specialties=["Internal Med", "Cardiology", "Orthopedics", "Neurology", "General Surgery"],
)

run_random_forest_and_score_data(df_name_prefix, claims_df)
