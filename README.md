# MLE Assignment - Predicting Vomitoxin Concentration

## Overview
This project focuses on predicting vomitoxin (DON) concentration in grains using Machine Learning models. The approach involves data preprocessing, feature selection, and model optimization using TPOT, ElasticNet, KNN, and RandomForest.

## Approach

### 1. **Data Preprocessing**
- Loaded the dataset and removed unnecessary columns (like `hsi_id`).
- Handled missing values by dropping or imputing them with mean values.
- Standardized features using `StandardScaler` to ensure uniformity.
- Visualized the target variable (`vomitoxin_ppb`) distribution to understand the data better.

### 2. **Feature Engineering & Selection**
- Removed outliers using the IQR method.
- Applied PCA for dimensionality reduction (retaining 95% variance).
- Used Recursive Feature Elimination (RFE) with ElasticNet to select the most important features.
- Ranked feature importance using ElasticNet coefficients.

### 3. **Model Training & Optimization**
#### **Initial Approach - TPOT AutoML**
- Started with `TPOTRegressor` for automated model selection and hyperparameter tuning.
- Configured TPOT with `neg_mean_absolute_error` as the scoring metric to optimize for lower MAE.
- Allowed 10 generations of evolution and a population size of 100 to explore a variety of models.
- TPOT initially suggested tree-based models, but the performance was suboptimal, with an R² score around **0.61**.

#### **Switching to Manual Model Selection**
- Built a pipeline with ElasticNetCV for feature selection and KNN as the regressor.
- Tuned hyperparameters using `GridSearchCV`, leading to an improvement in performance with **R² = 0.68**.

#### **Enhancing Performance - Stacked ElasticNet & RandomForest**
- Trained an ElasticNet model first to get predictions.
- Used these predictions as features for a RandomForest model.
- Introduced data augmentation by adding Gaussian noise to the training data.
- This hybrid approach improved the R² score significantly to **0.76**, with a lower MAE and MSE compared to previous methods.

### 4. **Model Evaluation**
For each model, evaluated performance using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-Squared (R²)**

#### **Final Results Comparison**
| Model                           | MAE  | MSE  | R²  |
|--------------------------------|------|------|------|
| TPOT Best Model               | 5.12 | 32.4 | 0.61 |
| ElasticNet + KNN               | 4.35 | 26.7 | 0.68 |
| ElasticNet + RandomForest      | 3.21 | 18.9 | 0.76 |

### 5. **Final Thoughts**
- TPOT provided a strong baseline but required additional tuning.
- Combining ElasticNet with KNN and RandomForest improved generalization.
- Feature selection and outlier removal significantly impacted model performance.
- Data augmentation helped in training a more robust model.

## Next Steps
- Experiment with deep learning models (e.g., Neural Networks).
- Explore other AutoML frameworks like H2O.ai.
- Implement real-time prediction APIs.

## Running the Code
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python MLE Assignment.py`
3. Check the output for best model details and evaluation metrics.

---
This project was a great learning experience in balancing automation (TPOT) with manual tuning for optimal results!

