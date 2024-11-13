# RIS Ranking POC

A proof of concept for ranking medical claims using synthetic data and machine learning.

## Setup and Installation

1. Install Poetry (package manager):
   
   **Windows (PowerShell):**
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```

   **macOS/Linux:**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone this repository:
   ```bash
   git clone <repository-url>
   cd ris-ranking-poc
   ```

3. Install project dependencies:
   ```bash
   poetry install
   ```

4. Run the application in one of two ways:

   **Option 1:** Activate virtual environment first
   ```bash
   poetry shell
   python main.py
   ```

   **Option 2:** Run directly through Poetry
   ```bash
   poetry run python main.py
   ```

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

## Package Management

This project uses Poetry for dependency management. Common Poetry commands:

- Add a new package:
  ```bash
  poetry add package_name
  ```

- Remove a package:
  ```bash
  poetry remove package_name
  ```

- Update all dependencies:
  ```bash
  poetry update
  ```

- Show currently installed packages:
  ```bash
  poetry show
  ```

- Export dependencies to requirements.txt:
  ```bash
  poetry export -f requirements.txt --output requirements.txt
  ```
