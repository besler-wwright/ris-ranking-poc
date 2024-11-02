# RIS Ranking POC

A proof of concept for ranking medical claims using synthetic data and machine learning.

## Features

- Generates synthetic medical claims data with configurable parameters
- Uses Random Forest model to predict claim priorities
- Calculates priority scores based on multiple factors
- Supports customizable number of providers, diagnosis codes, and procedure codes

## Key Components

- `randomforest.py`: Main implementation with synthetic data generation and model training
- Includes functions for:
  - Synthetic claims generation
  - Feature preparation
  - Model training and evaluation
  - Priority score calculation

## Usage

The system generates synthetic medical claims and uses a Random Forest model to predict priorities based on various features including:
- Length of stay
- Claim charges
- Provider information
- Diagnosis codes
- Procedure codes

## Dependencies

Managed via Poetry. Key dependencies include:
- NumPy
- Pandas
- Scikit-learn

## Setup

1. Ensure Poetry is installed
2. Run `poetry install`
3. Use the provided functions to generate data and train models
