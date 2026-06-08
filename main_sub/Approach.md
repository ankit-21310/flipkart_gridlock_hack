# Gridlock Hackathon 2.0 Approach

This document outlines the machine learning approach, feature engineering, and ensembling strategy used to predict demand for the Flipkart Grid ML challenge.

## 1. Pipeline Overview
The pipeline is divided into two separate scripts to separate training and inference:
* **`train.py`**: Handles data preprocessing, K-Fold cross-validation training, ensemble weight optimization, and saves all models and preprocessing artifacts.
* **`infer.py`**: Loads the pre-trained models and artifacts, applies the exact same preprocessing to the unseen test dataset, and generates the final predictions (`submission_final.csv`).

## 2. Feature Engineering & Preprocessing
The preprocessing logic is specifically designed to handle localized, time-dependent variations in features like Weather and Temperature.

### Time-based Features
* **Timestamp Parsing**: The `timestamp` column is split into discrete `hour` and `minute` features.
* **Cyclical Encoding**: Since time is continuous and cyclical, the time (in minutes from midnight) is transformed into `sin_time` and `cos_time` using sine and cosine functions. This helps the models understand the cyclical nature of a 24-hour day.

### Missing Value Imputation
* **RoadType**: Imputed using the global mode from the training dataset.
* **Temperature**: Missing values are imputed hierarchically to preserve local and temporal contexts:
  1. Median temperature for the specific `geohash` and `hour`.
  2. Fallback to median temperature for the specific `day` and `hour`.
  3. Global fallback to the overall training dataset median temperature.
* **Weather**: Similar hierarchical imputation using modes:
  1. Mode weather condition for the specific `geohash`.
  2. Fallback to mode weather for the specific `day`.
  3. Global fallback to `'Sunny'`.

### Categorical Encoding
Categorical features (`geohash`, `RoadType`, `LargeVehicles`, `Landmarks`, `Weather`) are encoded using `LabelEncoder`. The encoders are fitted on the combined train and test sets to ensure consistent mapping of categories, then saved as artifacts for inference.

## 3. Modeling Strategy

We employ a robust ensemble of Gradient Boosting Decision Trees. 
* **Algorithms**: **LightGBM**, **XGBoost**, and **CatBoost**. Each algorithm possesses unique strengths in handling tabular data and categorical splits.
* **Cross-Validation**: We utilize **5-Fold Cross-Validation** (`KFold`). This prevents overfitting and allows us to generate reliable Out-Of-Fold (OOF) predictions.
* **Parameters**: 
  * LightGBM and XGBoost parameters are set to a low learning rate (`0.03`) with a moderate depth (`8`) and feature fraction (`0.8`) to ensure steady convergence. 
  * CatBoost leverages its built-in advanced categorical encoding explicitly with the `cat_features` argument.
  * (Optional) An Optuna study is included in the training pipeline to tune hyper-parameters dynamically based on a validation split.

## 4. Ensembling via Optuna
Rather than taking a simple geometric or arithmetic mean, we optimize the weights of the models to minimize the Root Mean Squared Error (RMSE).
1. During the 5-Fold cross-validation, OOF predictions from all three model types are collected.
2. An **Optuna** study searches for the optimal weights ($w_{lgb}$, $w_{xgb}$, $w_{cb}$) constrained such that they sum to 1.0.
3. The objective function calculates the RMSE between the true target and the weighted linear combination of the OOF predictions.
4. The best weights are saved as an artifact.

## 5. Inference
During inference (`infer.py`):
1. **Artifact Loading**: Preprocessing statistics (medians, modes) and `LabelEncoders` are loaded.
2. **Preprocessing**: Test data goes through the exact same transformations as the training data using the saved statistics.
3. **Prediction Generation**: 
   * The test set is evaluated against all 5 fold-models for LightGBM, XGBoost, and CatBoost. 
   * The predictions are averaged across the 5 folds for each algorithm.
4. **Final Assembly**: The distinct algorithm predictions are multiplied by their respective Optuna-optimized weights to yield the final `demand` predictions.

## 6. How to Run
1. **Train**:
   ```bash
   python train.py
   ```
   *This will generate a `models/` directory containing all `.pkl` files (artifacts and models).*

2. **Infer**:
   ```bash
   python infer.py
   ```
   *This will output `submission_final.csv`.*
