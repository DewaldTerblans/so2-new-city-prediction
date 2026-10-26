# SO2 New-City Prediction Project

Statistical and machine learning models for predicting quarterly atmospheric sulfur dioxide ($\text{SO}_2$) levels across new, unseen US cities.

## Overview

This project evaluates multiple predictive models—ranging from classical Multiple Linear Regression (MLR) and regularized regressions (Ridge/Lasso) to Random Forest and Linear Mixed-Effects models. The core challenge is evaluating how accurately these models generalize to entirely new geographical locations (cities not included during training).

### Key Results
* **New-City RMSE:** Drops from a benchmark of **~23.98** down to **~15–16** with tuned ensemble and mixed-effects models.
* **Mixed-Effects Performance:** Outperforms standalone regression techniques once at least one quarterly observation from a new city is available to estimate city-specific random effects.

---

## Dataset

The dataset (`quarterly_SO2_data.csv`) contains quarterly environmental and meteorological observations across 41 US cities collected from 1969 to 1971.

* **Location:** `data/quarterly_SO2_data.csv`

---

## Repository Structure

```text
so2-new-city-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── quarterly_SO2_data.csv
├── src/
│   ├── so2_eval.py
│   ├── so2_mixed_effects.py
│   ├── so2_random_forest.py
│   ├── so2_worked_example.py
│   ├── mlr.py
│   └── ridge_lasso.py
└── docs/
    ├── SO2_Project_Report.pdf
    └── SO2_project_presentation.pdf
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/so2-new-city-prediction.git
   cd so2-new-city-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Models

Run the scripts from the root directory by specifying the path to the CSV file:

* **Evaluation Pipeline:**
  ```bash
  python src/so2_eval.py data/quarterly_SO2_data.csv
  ```

* **Mixed-Effects Model:**
  ```bash
  python src/so2_mixed_effects.py data/quarterly_SO2_data.csv
  ```

* **Random Forest Model:**
  ```bash
  python src/so2_random_forest.py data/quarterly_SO2_data.csv rf_output
  ```

* **Worked Example (Specific City):**
  ```bash
  python src/so2_worked_example.py data/quarterly_SO2_data.csv "San Francisco"
  ```

* **Multiple Linear Regression (MLR):**
  ```bash
  python src/mlr.py
  ```

* **Ridge & Lasso Regression:**
  ```bash
  python src/ridge_lasso.py
  ```