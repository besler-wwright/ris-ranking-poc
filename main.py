import csv
import os

from loguru import logger
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


def generate_synthetic_claims(
    df_name_prefix="MyData",
    num_claims=1000,
    seed=42,
    num_of_payors=5,
    num_of_providers=50,
    num_of_diagnosis_codes=30,
    num_of_procedure_codes=40,
    specialties=["Internal Med", "Cardiology", "Orthopedics", "Neurology", "General Surgery"],
    should_display_stats=False,
):
    """
    Generate synthetic medical claims data with realistic patterns.
    """

    c.print(f"\n[bold green]Generating {df_name_prefix} dataset...[/bold green]")
    np.random.seed(seed)

    csv_filename = f"data/{df_name_prefix}__synthesized_medical_claims.csv"

    # Try to load existing file
    if os.path.exists(csv_filename):
        existing_df = pd.read_csv(csv_filename)
        if len(existing_df) == num_claims:
            c.print(f"\t[yellow]Loading existing dataset of {num_claims} claims from {csv_filename}[/yellow]")
            # Convert date_submitted back to datetime
            existing_df["date_submitted"] = pd.to_datetime(existing_df["date_submitted"])
            return existing_df
        else:
            c.print(f"\t[yellow]Existing dataset has different number of claims ({len(existing_df)} vs {num_claims}). Generating BRAND NEW dataset[/yellow]")

    # Create lists for categorical variables
    provider_ids = [f"PRV{str(i).zfill(4)}" for i in range(1, num_of_providers + 1)]  # 50 providers
    payor_ids = [f"PAY{str(i).zfill(3)}" for i in range(1, num_of_payors + 1)]  # 5 payors
    diagnosis_codes = [f"ICD{str(i).zfill(3)}" for i in range(1, num_of_diagnosis_codes + 1)]  # 30 diagnosis codes
    procedure_codes = [f"CPT{str(i).zfill(4)}" for i in range(1, num_of_procedure_codes + 1)]  # 40 procedure codes

    # Generate diagnosis-specific average LOS
    diagnosis_los = {}
    # Calculate the size of each third
    third_size = len(diagnosis_codes) // 3
    for i, dx_code in enumerate(diagnosis_codes):
        # First third: complex conditions
        if i < third_size:
            diagnosis_los[dx_code] = np.round(np.random.uniform(5, 10), 1)
        # Second third: moderate conditions
        elif i < third_size * 2:
            diagnosis_los[dx_code] = np.round(np.random.uniform(3, 6), 1)
        # Last third: less complex conditions
        else:
            diagnosis_los[dx_code] = np.round(np.random.uniform(1, 4), 1)

    # Generate procedure-diagnosis-payor combination specific charges
    proc_dx_charges = {}
    for proc_code in procedure_codes:
        for dx_code in diagnosis_codes:
            for payor_id in payor_ids:
                base_charge = np.random.uniform(1000, 9000)
                multiplier = np.random.uniform(0.8, 1 + (num_of_payors * 0.1))
                proc_dx_charges[(proc_code, dx_code, payor_id)] = round(base_charge * multiplier, 2)

    procedure_code_list = np.random.choice(procedure_codes, num_claims)
    dx_code_list = np.random.choice(diagnosis_codes, num_claims)
    payer_id_list = np.random.choice(payor_ids, num_claims)

    # Generate base data
    data = {
        "claim_id": [f"CLM{str(i).zfill(6)}" for i in range(1, num_claims + 1)],
        "date_submitted": [(datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime("%Y-%m-%d") for _ in range(num_claims)],
        "payor_id": np.random.choice(payor_ids, num_claims),
        "provider_id": np.random.choice(provider_ids, num_claims),
        "provider_specialty": np.random.choice(specialties, num_claims),
        "diagnosis_code": np.random.choice(diagnosis_codes, num_claims),
        "procedure_code": procedure_code_list,
        "claim_charges": [proc_dx_charges[(p_code, d_code, p_id)] for p_code, d_code, p_id in zip(procedure_code_list, dx_code_list, payer_id_list)],
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
    # Base probability
    if num_of_diagnosis_codes < 2:
        base = np.random.random(num_claims) * 0.1
    else:
        base = np.random.random(num_claims) * 0.02
    rework_probabilities = (
        base
        +
        # Higher amounts more likely to need rework
        (df["claim_charges"] > 2000).astype(float) * 0.1
        +
        # Certain procedures more likely to need rework
        (df["procedure_code"].isin(["CPT0001", "CPT0003", "CPT0005"])).astype(float) * 0.15
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
        # np.random.uniform(-0.3, 0.3, size=rework_mask.sum()) +
        df.loc[rework_mask, "los_difference"]
        * 0.1
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
        "payor_id",
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
    df.to_csv(csv_filename, index=False)
    c.print(f"\t[green]{df_name_prefix} saved to {csv_filename}[/green]")

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
    data["payor_id_encoded"] = le.fit_transform(data["payor_id"])
    data["provider_id_encoded"] = le.fit_transform(data["provider_id"])
    data["procedure_code_encoded"] = le.fit_transform(data["procedure_code"])
    data["diagnosis_code_encoded"] = le.fit_transform(data["diagnosis_code"])
    data["provider_specialty_encoded"] = le.fit_transform(data["provider_specialty"])

    # Create feature list for model
    features = [
        # "claim_charges",
        "payor_id_encoded",
        "provider_id_encoded",
        "procedure_code_encoded",
        "diagnosis_code_encoded",
        "provider_specialty_encoded",
        # "avg_los",
        # "actual_los",
        "los_difference",
    ]

    return data, features


def display_stats_for_df(df):
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
    c.print(df.head())

    # c.print("\nSummary statistics:")
    # c.print(claims_df.describe())

    c.print("\n[black]-----LOS statistics by procedure type:[/black]--------------------------------")
    c.print(df.groupby("procedure_code")[["avg_los", "actual_los", "los_difference"]].mean())

    c.print("\n[black]-----Correlation between LOS difference and rework:[/black]-------------------")
    c.print(df["los_difference"].abs().corr(df["needs_rework"]))


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


def train_and_evaluate_random_forest_model(data, features):
    """
    Train a machine learning model to predict which claims need rework and evaluate how well it performs.
    
    This function:
    1. Prepares the data for training
    2. Splits data into training and testing sets
    3. Normalizes the data
    4. Trains a Random Forest model
    5. Evaluates the model's performance
    """
    # X contains our input features, y contains what we're trying to predict (needs_rework)
    X = data[features]
    y = data["needs_rework"]

    # Make sure we have both claims that need rework (1) and don't need rework (0)
    # The model needs examples of both to learn effectively
    if len(np.unique(y)) < 2:
        raise ValueError("Target variable 'needs_rework' has only one class. Need both positive and negative examples for training.")

    # Split our data into two parts:
    # - 80% for training (teaching the model)
    # - 20% for testing (checking how well it learned)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # StandardScaler makes sure all our features are on the same scale
    # (like converting feet and inches both to meters)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the Random Forest model
    # Random Forest works by creating many decision trees and combining their predictions
    # It's like getting opinions from multiple experts and taking a vote
    model = RandomForestClassifier(
        n_estimators=100,  # Create 100 different decision trees
        max_depth=None,    # Let trees grow as deep as needed
        min_samples_split=2,  # Minimum samples needed to split a node
        min_samples_leaf=1,   # Minimum samples needed in a leaf node
        random_state=42       # For reproducible results
    )
    # Train the model using our training data
    model.fit(X_train_scaled, y_train)

    # Calculate how important each feature was in making predictions
    # Higher importance means that feature was more useful in predicting rework needs
    feature_importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    feature_importance = feature_importance[feature_importance["importance"] > 0].sort_values("importance", ascending=False)

    # Get probability predictions for our test data
    # This tells us how confident the model is that each claim needs rework
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Get probability of needs_rework=1

    return model, scaler, feature_importance, X_test, y_test, y_pred_proba


def score_random_forest_data(df_name_prefix, claims_df, rework_prob, impact_score, priority_score):
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
    csv_filename = f"data/{df_name_prefix}_random_forest_scored_medical_claims.csv"
    scored_claims_sorted.to_csv(csv_filename, index=False)
    c.print(f"\t[green]{df_name_prefix} scored data saved to {csv_filename}[/green]")
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
    scored_claims_sorted = score_random_forest_data(df_name_prefix, claims_df, rework_prob, impact_score, priority_score)

    # Print summary statistics and top priority claims
    c.print("\n[bold green]-----Feature Importance:[/bold green]--------------------------------")
    c.print(feature_importance)

    c.print("\n[black]-----Top 10 Priority Claims:[/black]--------------------------------")
    c.print(
        scored_claims_sorted[
            ["impact_score", "claim_id", "diagnosis_code", "procedure_code", "claim_charges", "rework_probability", "priority_score", "los_difference", "payment_difference"]
        ].head(  # noqa
            10
        )
    )


@logger.catch
def run_simple_scenario():
    c = Console()
    df_name_prefix = "SIMPLE"
    c.print(f"\n[bold green]Running {df_name_prefix} scenario *************************************[/bold green]")
    c.print("\t[bold cyan]Expecting LOS and Payors to be important features[/bold cyan]")
    claims_df = generate_synthetic_claims(
        df_name_prefix,
        num_claims=100,
        seed=42,
        num_of_payors=2,
        num_of_providers=1,
        num_of_diagnosis_codes=1,
        num_of_procedure_codes=1,
        specialties=["Neurology"],
    )
    run_random_forest_and_score_data(df_name_prefix, claims_df)
    c.print(f"\n[bold green]End of {df_name_prefix} scenario *************************************[/bold green]")


@logger.catch
def run_less_simple_scenario():
    c = Console()
    df_name_prefix = "LESS_SIMPLE"
    c.print(f"\n[bold green]Running {df_name_prefix} scenario *************************************[/bold green]")
    c.print("\t[bold cyan]Expecting LOS, Payors, and Provider Specialties to be important features[/bold cyan]")
    claims_df = generate_synthetic_claims(
        df_name_prefix,
        num_claims=100,
        seed=42,
        num_of_payors=2,
        num_of_providers=1,
        num_of_diagnosis_codes=1,
        num_of_procedure_codes=1,
        specialties=["Internal Med", "Cardiology", "Orthopedics", "Neurology", "General Surgery"],
    )
    run_random_forest_and_score_data(df_name_prefix, claims_df)
    c.print(f"\n[bold green]End of {df_name_prefix} scenario *************************************[/bold green]")


@logger.catch
def run_even_less_simple_scenario():
    c = Console()
    df_name_prefix = "EVEN_LESS_SIMPLE"
    c.print(f"\n[bold green]Running {df_name_prefix} scenario *************************************[/bold green]")
    c.print("\t[bold cyan]Expecting LOS, Diag Codes, and Payors to be important features[/bold cyan]")
    claims_df = generate_synthetic_claims(
        df_name_prefix,
        num_claims=100,
        seed=42,
        num_of_payors=2,
        num_of_providers=1,
        num_of_diagnosis_codes=10,
        num_of_procedure_codes=1,
        specialties=["General Surgery"],
    )
    run_random_forest_and_score_data(df_name_prefix, claims_df)
    c.print(f"\n[bold green]End of {df_name_prefix} scenario *************************************[/bold green]")


@logger.catch
def run_standard_scenario_01():
    c = Console()
    df_name_prefix = "STANDARD_SCENARIO_01"
    c.print(f"\n[bold green]Running {df_name_prefix} scenario *************************************[/bold green]")
    c.print("\t[bold cyan]Expecting LOS, Diag Codes, and Payors to be important features[/bold cyan]")
    claims_df = generate_synthetic_claims(
        df_name_prefix,
        num_claims=1000,
        seed=42,
        num_of_payors=10,
        num_of_providers=5,
        num_of_diagnosis_codes=10,
        num_of_procedure_codes=19,
        specialties=["General Surgery"],
    )
    run_random_forest_and_score_data(df_name_prefix, claims_df)
    c.print(f"\n[bold green]End of {df_name_prefix} scenario *************************************[/bold green]")


if __name__ == "__main__":

    # Start ---------------------------------------
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # run_simple_scenario()
    # run_less_simple_scenario()
    run_even_less_simple_scenario()
    # run_standard_scenario_01()
