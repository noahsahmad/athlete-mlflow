# athlete-mlflow
# README: End-to-End ML Pipeline for CrossFit Athlete Deadlift Prediction

## Project Overview
This project implements an end-to-end Machine Learning pipeline to predict a CrossFit athlete's deadlift performance based on their physical attributes (age, height, weight) and engineered features. The pipeline utilizes **Databricks**, **MLflow**, and the **Databricks Feature Store** to manage the lifecycle of data, features, and model experiments.

## Prerequisites & Environment Setup
1. **Databricks Workspace:** A Databricks environment is required to execute this notebook.
2. **Cluster Runtime:** Use a **Databricks Machine Learning Runtime** (e.g., **14.3 LTS ML** or **15.4 LTS ML**). 
   * *Note: Avoid DBR 18.x or Serverless compute if you encounter `InvalidVersion` errors related to the Feature Store client.* 
3. **Data Upload:** Upload the `athletes.csv` dataset to your Databricks workspace (e.g., to DBFS at `/dbfs/FileStore/tables/athletes.csv` or a Unity Catalog Volume).
4. **Libraries:** The Databricks ML runtime includes most required libraries (`mlflow`, `scikit-learn`, `pandas`, `pyspark`). If using a standard cluster, ensure you install `databricks-feature-store` and `mlflow` via `%pip install`.

## Execution Instructions
Follow these steps to execute the pipeline sequentially from top to bottom:

### Step 1: Data Ingestion & Setup
* Locate the **Dataset Selection and Setup** section.
* Ensure the file path in `pd.read_csv(...)` correctly points to your uploaded `athletes.csv` location.
* Run the initial EDA cells to profile the raw dataset.

### Step 2: Data Cleaning & Preprocessing
* Execute the cleaning cells to replace extreme physical outliers (e.g., negative weights, impossible ages/heights, deadlifts > 1200 lbs) with `NaN`.
* The code will automatically handle the conversion of multi-select string columns (e.g., `experience`, `schedule`) into one-hot encoded dummy variables.

### Step 3: Feature Store Registration
* Run the **Feature Store Integration** cells.
* The code establishes two Feature Tables:
  * `athletes_features_v1`: Baseline features (age, height, weight).
  * `athletes_features_v2`: Enhanced features including engineered `BMI` and cleaned column names.
* *Note: If a table already exists, you may need to drop it via the Databricks UI or rename the table variables in the code.* 

### Step 4: Model Training & Experimentation
* Navigate to the **Experimentation and Model Training** section.
* Run the nested loop cell. This will automatically train `RandomForestRegressor` models across 4 combinations:
  * **Data:** Feature Version 1 vs. Feature Version 2.
  * **Hyperparameters:** Shallow Tree (`n_estimators=50`, `max_depth=5`) vs. Deeper Tree (`n_estimators=100`, `max_depth=10`).
* All parameters, features, and evaluation metrics (RMSE, R2) are automatically logged to **MLflow**.
* The models are logged using `fs.log_model`, inherently tying them to their Feature Store lineage.

### Step 5: Viewing Results
* Click the **Experiment** flask icon 🧪 in the top right of your Databricks notebook (or click the MLflow links generated in the cell outputs).
* You can view and compare the 4 runs side-by-side to analyze how the engineered `BMI` feature and deeper tree hyperparameters affected the model's R-squared and RMSE metrics.
