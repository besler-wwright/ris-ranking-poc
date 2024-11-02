# RIS Ranking POC

A proof of concept for ranking medical claims using synthetic data and machine learning.

## Features

- Generates synthetic medical claims data with configurable parameters
- Uses Random Forest model to predict claim priorities
- Calculates priority scores based on multiple factors
- Supports customizable number of providers, diagnosis codes, and procedure codes
- Persists generated datasets and reuses them when parameters match

## Key Components

- `main.py`: Main implementation with synthetic data generation and model training
- Includes functions for:
  - Synthetic claims generation with data persistence
  - Feature preparation
  - Model training and evaluation
  - Priority score calculation

## Data Management

The system automatically saves generated datasets to the `data/` directory using the format:
- `{df_name_prefix}__synthesized_medical_claims.csv` for raw data
- `{df_name_prefix}_random_forest_scored_medical_claims.csv` for scored data

When generating data, the system will:
1. Check for existing datasets
2. Reuse matching datasets if they exist (same number of claims)
3. Generate new data only when necessary

## Usage

The system generates synthetic medical claims and uses a Random Forest model to predict priorities based on various features including:
- Length of stay
- Claim charges
- Provider information
- Diagnosis codes
- Procedure codes
- Payor information

## Dependencies

Managed via Poetry. Key dependencies include:
- NumPy
- Pandas
- Scikit-learn
- Rich (for console output)
- Loguru (for logging)

## Setup

1. Ensure Poetry is installed
2. Run `poetry install`
3. Use the provided functions to generate data and train models
