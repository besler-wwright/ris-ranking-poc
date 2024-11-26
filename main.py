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
    Creates realistic-looking medical claims data for testing our system.
    This is like creating a practice dataset that mimics real medical claims
    so we can test our analysis methods safely.

    The function creates claims with:
    - Unique claim IDs and dates
    - Insurance companies (payors) and healthcare providers
    - Medical diagnoses and procedures
    - Length of stay information
    - Cost and payment details
    """

    c.print(f"\n[bold green]Generating {df_name_prefix} dataset...[/bold green]")
    # Set a random seed for consistent results
    # Like using the same recipe each time to bake a cake
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

    # Create IDs for different parts of the healthcare system
    provider_ids = [f"PRV{str(i).zfill(4)}" for i in range(1, num_of_providers + 1)]  # Doctor/Hospital IDs
    payor_ids = [f"PAY{str(i).zfill(3)}" for i in range(1, num_of_payors + 1)]  # Insurance company IDs
    diagnosis_codes = [f"ICD{str(i).zfill(3)}" for i in range(1, num_of_diagnosis_codes + 1)]  # Disease codes
    procedure_codes = [f"CPT{str(i).zfill(4)}" for i in range(1, num_of_procedure_codes + 1)]  # Treatment codes

    # Create typical hospital stay lengths for different diagnoses
    # Some conditions need longer stays than others
    diagnosis_los = {}
    third_size = len(diagnosis_codes) // 3
    for i, dx_code in enumerate(diagnosis_codes):
        # Complex conditions (like major surgeries): 5-10 days
        if i < third_size:
            diagnosis_los[dx_code] = np.round(np.random.uniform(5, 10), 1)
        # Moderate conditions (like infections): 3-6 days
        elif i < third_size * 2:
            diagnosis_los[dx_code] = np.round(np.random.uniform(3, 6), 1)
        # Simple conditions (like minor procedures): 1-4 days
        else:
            diagnosis_los[dx_code] = np.round(np.random.uniform(1, 4), 1)

    # Create typical costs for different procedure-diagnosis-insurance combinations
    # Different insurance companies might pay different amounts for the same procedure
    proc_dx_charges = {}
    for proc_code in procedure_codes:
        for dx_code in diagnosis_codes:
            for payor_id in payor_ids:
                base_charge = np.random.uniform(1000, 9000)  # Base cost between $1,000 and $9,000
                multiplier = np.random.uniform(0.8, 1 + (num_of_payors * 0.1))  # Different insurers pay different amounts
                proc_dx_charges[(proc_code, dx_code, payor_id)] = round(base_charge * multiplier, 2)

    procedure_code_list = np.random.choice(procedure_codes, num_claims)
    dx_code_list = np.random.choice(diagnosis_codes, num_claims)
    payer_id_list = np.random.choice(payor_ids, num_claims)

    # Create the basic claim information
    data = {
        "claim_id": [f"CLM{str(i).zfill(6)}" for i in range(1, num_claims + 1)],  # Unique claim numbers
        "date_submitted": [(datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime("%Y-%m-%d") for _ in range(num_claims)],  # Random dates in 2024
        "payor_id": np.random.choice(payor_ids, num_claims),  # Random insurance company
        "provider_id": np.random.choice(provider_ids, num_claims),  # Random healthcare provider
        "provider_specialty": np.random.choice(specialties, num_claims),  # Random doctor specialty
        "diagnosis_code": np.random.choice(diagnosis_codes, num_claims),  # Random diagnosis
        "procedure_code": procedure_code_list,  # Random procedure
        "claim_charges": [proc_dx_charges[(p_code, d_code, p_id)] for p_code, d_code, p_id in zip(procedure_code_list, dx_code_list, payer_id_list)],  # Look up the cost
    }

    # Convert our data into a DataFrame (like a spreadsheet)
    df = pd.DataFrame(data)

    # Add information about hospital stays
    df["avg_los"] = df["diagnosis_code"].map(diagnosis_los)  # Expected stay length for each diagnosis

    # Create actual length of stay with some natural variation
    # Real stays might be shorter or longer than expected
    df["actual_los"] = df.apply(lambda row: max(1, np.random.normal(row["avg_los"], row["avg_los"] * 0.2)), axis=1).round(1)

    # Calculate how different the actual stay was from expected
    df["los_difference"] = (df["actual_los"] - df["avg_los"]).round(1)

    # Determine which claims need rework based on various risk factors
    # More complex cases have higher chances of needing review
    base = np.random.random(num_claims) * (0.02 if num_of_diagnosis_codes >= 2 else 0.1)
    rework_probabilities = (
        base
        + (df["claim_charges"] > 2000).astype(float) * 0.1  # Higher costs increase risk
        + (df["procedure_code"].isin(["CPT0001", "CPT0003", "CPT0005"])).astype(float) * 0.15  # Certain procedures are riskier
        + (df["provider_id"].isin(["PRV0001", "PRV0002"])).astype(float) * 0.2  # Some providers have higher error rates
        + (abs(df["los_difference"]) > 2).astype(float) * 0.2  # Unusual stay lengths increase risk
    )

    # Mark claims as needing rework if their risk is high enough
    df["needs_rework"] = (rework_probabilities > 0.5).astype(int)

    # For claims that need rework, calculate potential payment changes
    # This shows how much money might be affected
    df["payment_difference"] = 0.0
    rework_mask = df["needs_rework"] == 1
    df.loc[rework_mask, "payment_difference"] = df.loc[rework_mask, "claim_charges"] * (df.loc[rework_mask, "los_difference"] * 0.1)

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
    Converts raw claims data into a format that our machine learning model can understand.

    The model needs numbers to work with, so we need to:
    1. Convert text values (like provider IDs) into numbers
    2. Keep only the information that helps predict rework needs

    It's like translating a book into a language the computer can read.
    """
    # Make a fresh copy of our data to avoid changing the original
    data = df.copy()

    # Convert text categories into numbers using a label encoder
    # For example:
    # PAY001, PAY002, PAY003 becomes 0, 1, 2
    # This helps the model work with categorical data
    le = LabelEncoder()

    # Convert each text column into numbers
    data["payor_id_encoded"] = le.fit_transform(data["payor_id"])  # Insurance company IDs
    data["provider_id_encoded"] = le.fit_transform(data["provider_id"])  # Healthcare provider IDs
    data["procedure_code_encoded"] = le.fit_transform(data["procedure_code"])  # Medical procedure codes
    data["diagnosis_code_encoded"] = le.fit_transform(data["diagnosis_code"])  # Diagnosis codes
    data["provider_specialty_encoded"] = le.fit_transform(data["provider_specialty"])  # Doctor specialties

    # List of features we'll use to predict rework needs
    # We choose these based on what's most likely to affect rework:
    features = [
        # Who was involved
        "payor_id_encoded",  # Which insurance company
        "provider_id_encoded",  # Which healthcare provider
        "procedure_code_encoded",  # What procedure was done
        "diagnosis_code_encoded",  # What condition was treated
        "provider_specialty_encoded",  # Doctor's specialty
        # Time-related factors
        "los_difference",  # How different actual stay was from expected
    ]

    # Return both the prepared data and the list of features we'll use
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
    Calculate how important each claim is by combining two factors:
    1. How likely the claim needs rework
    2. How big an impact rework would have (in terms of time and money)

    This helps us focus on claims that both:
    - Have a high chance of needing rework
    - Would have a significant effect if reworked
    """
    # Prepare the data for our prediction model
    # First get just the features we care about
    X = data[features]
    # Then scale the numbers so they're all comparable
    # (like converting everything to the same unit of measurement)
    X_scaled = scaler.transform(X)

    # Ask our trained model how likely each claim needs rework
    # This gives us a probability from 0% to 100%
    rework_prob = model.predict_proba(X_scaled)[:, 1]

    # Calculate typical impacts we see in the data
    # This helps us understand what's "normal" vs "unusual"
    # avg_payment_impact = abs(data["payment_difference"]).mean()  # Average money impact
    # avg_los_impact = abs(data["los_difference"]).mean()         # Average time impact

    # Convert impacts to a 0-1 scale for fair comparison
    # (like grading on a curve from 0 to 100%)
    normalized_payment_impact = abs(data["payment_difference"]) / abs(data["payment_difference"]).max()
    normalized_los_impact = abs(data["los_difference"]) / abs(data["los_difference"]).max()

    # Calculate overall impact score
    # We weight money impact (70%) more than time impact (30%)
    # because financial impact is usually more critical
    impact_score = normalized_payment_impact * 0.7 + normalized_los_impact * 0.3

    # Calculate final priority score
    # We weight probability of rework (60%) slightly more than impact (40%)
    # because we want to focus on claims most likely to need attention
    priority_score = rework_prob * 0.6 + impact_score * 0.4

    # Return all three scores for each claim:
    # 1. How likely it needs rework
    # 2. How big an impact it might have
    # 3. Overall priority ranking
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
        max_depth=None,  # Let trees grow as deep as needed
        min_samples_split=2,  # Minimum samples needed to split a node
        min_samples_leaf=1,  # Minimum samples needed in a leaf node
        random_state=42,  # For reproducible results
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
    """
    Takes the predictions from our model and adds them to our claims data, then saves the results.

    This function:
    1. Adds prediction scores to each claim
    2. Sorts claims by priority (highest priority first)
    3. Organizes the data columns in a logical order
    4. Saves the scored results to a CSV file

    Parameters:
    - df_name_prefix: Name prefix for the output file
    - claims_df: Original claims data
    - rework_prob: Model's prediction of how likely each claim needs rework (0-1)
    - impact_score: How big an effect rework might have (based on money and time)
    - priority_score: Combined score of probability and impact (higher = more important)
    """
    # Make a copy of our claims data so we don't modify the original
    scored_claims = claims_df.copy()

    # Add our three prediction scores to each claim:
    # 1. How likely it needs rework
    # 2. How big an impact rework would have
    # 3. Overall priority score combining both factors
    scored_claims["rework_probability"] = rework_prob
    scored_claims["impact_score"] = impact_score
    scored_claims["priority_score"] = priority_score

    # Sort all claims by priority score, putting highest priority claims first
    scored_claims_sorted = scored_claims.sort_values("priority_score", ascending=False)

    # Arrange columns in a logical order for the output file
    # First show basic claim info, then details, then our prediction scores
    column_order = [
        # Basic claim identification
        "claim_id",
        "date_submitted",
        "diagnosis_code",
        "procedure_code",
        "claim_charges",
        # Length of stay information
        "avg_los",
        "actual_los",
        "los_difference",
        # Provider information
        "provider_id",
        "provider_specialty",
        # Outcome information
        "needs_rework",
        "payment_difference",
    ]
    # Add our prediction scores to the end of the column list
    scored_claims_sorted = scored_claims_sorted[column_order + ["rework_probability", "impact_score", "priority_score"]]

    # Save the scored claims to a CSV file for later use
    csv_filename = f"data/{df_name_prefix}_random_forest_scored_medical_claims.csv"
    scored_claims_sorted.to_csv(csv_filename, index=False)
    c.print(f"\t[green]{df_name_prefix} scored data saved to {csv_filename}[/green]")

    return scored_claims_sorted


def run_random_forest_model(claims_df, should_display_stats=False):
    """
    Analyzes claims data to predict which claims might need rework.

    This function:
    1. Prepares the claims data for analysis
    2. Trains a prediction model
    3. Calculates how likely each claim needs rework
    4. Determines the impact and priority of each claim
    5. Optionally shows how well the model performed
    """
    c = Console()

    # Convert raw claims data into a format our model can understand
    # (like translating text into numbers)
    prepared_data, features = prepare_features(claims_df)

    # Train the model and get back several important pieces:
    # - model: the trained prediction system
    # - scaler: tool to make sure all numbers are on the same scale
    # - feature_importance: shows which claim attributes matter most
    # - X_test, y_test: data used to check model accuracy
    # - y_pred_proba: how confident the model is about each prediction
    model, scaler, feature_importance, X_test, y_test, y_pred_proba = train_and_evaluate_random_forest_model(prepared_data, features)

    # Calculate three scores for each claim:
    # 1. rework_prob: How likely the claim needs rework (0-100%)
    # 2. impact_score: How much time/money could be affected
    # 3. priority_score: Combined importance based on probability and impact
    rework_prob, impact_score, priority_score = calculate_priority_scores(prepared_data, model, scaler, features)

    # Convert probability predictions into yes/no decisions
    # (if probability > 50%, predict the claim needs rework)
    predictions = (y_pred_proba > 0.5).astype(int)

    # If requested, show detailed statistics about how well the model performed
    if should_display_stats:
        # Show how many claims were predicted to need/not need rework
        c.print("\nClass distribution in predictions:", np.unique(predictions, return_counts=True))
        c.print("Class distribution in test set:", np.unique(y_test, return_counts=True))

        # Show detailed accuracy metrics
        c.print("\nModel Performance:")
        c.print(classification_report(y_test, predictions, zero_division=0))

    # Return everything needed to score and prioritize the claims
    return feature_importance, rework_prob, impact_score, priority_score


def create_feature_importance_pie_chart(df_name_prefix, feature_importance):
    """
    Creates a pie chart showing how important each feature was in predicting rework needs.

    Parameters:
    - df_name_prefix: Name used for saving the chart file
    - feature_importance: DataFrame containing feature names and their importance scores
    """
    # Create a new figure with a specific size
    plt.figure(figsize=(12, 8))

    # Create pie chart with direct labels
    # Use _ to ignore any additional return values
    patches, *_ = plt.pie(feature_importance["importance"], labels=[f"{imp:.1%}" for imp in feature_importance["importance"]], startangle=90)  # Show percentages on slices

    # Create legend labels with feature names and percentages
    legend_labels = [f"{feat} ({imp:.1%})" for feat, imp in zip(feature_importance["feature"], feature_importance["importance"])]

    # Add legend
    plt.legend(patches, legend_labels, title="Feature Importance", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))  # Position legend to the right of the pie chart

    # Add title
    plt.title(f"Feature Importance Distribution - {df_name_prefix}")

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the chart
    plt.savefig(f"plots/{df_name_prefix}_feature_importance_pie.png", bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def run_random_forest_and_score_data(df_name_prefix, claims_df):
    """
    Main function that processes claims data to identify which claims need attention.

    This function:
    1. Analyzes the claims data using machine learning
    2. Scores each claim based on risk and impact
    3. Saves the results to a file
    4. Shows which factors are most important
    5. Displays the top priority claims that need attention

    Parameters:
    - df_name_prefix: Name used for saving the output file
    - claims_df: The claims data to analyze
    """
    c = Console()

    # Step 1: Run the machine learning model to analyze the claims
    # This gives us:
    # - feature_importance: which factors matter most in predicting rework
    # - rework_prob: likelihood each claim needs rework
    # - impact_score: potential effect on time/money
    # - priority_score: overall importance ranking
    feature_importance, rework_prob, impact_score, priority_score = run_random_forest_model(claims_df)

    # Step 2: Add our prediction scores to the claims data and sort by priority
    scored_claims_sorted = score_random_forest_data(df_name_prefix, claims_df, rework_prob, impact_score, priority_score)

    # Step 3: Show which factors were most important in making predictions
    # Higher importance means that factor was more useful in spotting claims that need rework
    c.print("\n[bold green]-----Feature Importance:[/bold green]--------------------------------")
    c.print(feature_importance)

    # Create pie chart visualization
    create_feature_importance_pie_chart(df_name_prefix, feature_importance)

    # Step 4: Display the 10 claims that need the most attention
    # These are sorted by priority_score, with highest priority first
    c.print("\n[black]-----Top 10 Priority Claims:[/black]--------------------------------")
    # Show key information about each high-priority claim
    c.print(
        scored_claims_sorted[
            [
                "priority_score",  # Overall importance ranking
                "impact_score",  # How big an effect rework might have
                "rework_probability",  # How likely it needs rework
                "claim_id",  # Unique identifier for the claim
                "diagnosis_code",  # What condition was treated
                "procedure_code",  # What procedure was performed
                "claim_charges",  # How much the claim cost
                "los_difference",  # Difference in length of stay
                "payment_difference",  # Potential payment impact
            ]
        ].head(10)
    )


@logger.catch
def run_simple_scenario():
    c = Console()
    df_name_prefix = "SIMPLE"
    c.print(f"\n[bold green]Running {df_name_prefix} scenario *************************************[/bold green]")
    c.print("\t[bold cyan]Expecting LOS and Payors to be important features[/bold cyan]")
    claims_df = generate_synthetic_claims(
        df_name_prefix,
        num_claims=1000,
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
    c.print("\t[bold cyan]Expecting Proc Codes, LOS, Diag Codes, and Payors to be important features[/bold cyan]")
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
    os.makedirs("plots", exist_ok=True)  # future charts and graphs

    run_simple_scenario()
    # run_less_simple_scenario()
    # run_even_less_simple_scenario()
    # run_standard_scenario_01()
