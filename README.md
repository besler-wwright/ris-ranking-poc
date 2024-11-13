# RIS Ranking POC

A proof of concept for ranking medical claims using synthetic data and machine learning.

## Setup and Installation

### Prerequisites

- Python ^3.10 (version 3.10 or higher)
- Poetry package manager

### Installation Steps

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

   This will install all required packages:
   - pandas 2.2.3
   - numpy 2.1.2
   - scikit-learn 1.5.2
   - matplotlib 3.9.2
   - rich 13.9.4
   - loguru 0.7.2

4. Run the application in one of two ways:

   **Option 1:** Activate virtual environment first
   ```bash
   poetry shell
   python main.py
   ```

   **Option 2:** Run directly through Poetry
   ```bash
   poetry run start
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

This project uses Poetry for dependency management. The dependencies are locked in poetry.lock to ensure consistent installations across environments. Key Poetry commands:

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

- Check dependency conflicts:
  ```bash
  poetry check
  ```

- List outdated packages:
  ```bash
  poetry show --outdated
  ```

Note: This project requires Python 3.10 or higher. All dependencies are locked to specific versions in poetry.lock for reproducibility.
