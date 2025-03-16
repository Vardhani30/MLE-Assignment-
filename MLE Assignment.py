#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tpot


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tpot import TPOTRegressor

# Load dataset
file_path = "D:\Downloads\MLE-Assignment.csv"  # Update this with your actual file path
df = pd.read_csv(file_path)

# Remove the identifier column
df.drop(columns=['hsi_id'], inplace=True)

# Check for missing values
if df.isnull().sum().sum() > 0:
    df.dropna(inplace=True)  # Drop rows with missing values

# Plot target variable distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['vomitoxin_ppb'], bins=30, kde=True)
plt.title('Distribution of DON Concentration (vomitoxin_ppb)')
plt.xlabel('vomitoxin_ppb')
plt.ylabel('Frequency')
plt.show()

# Split features and target
X = df.drop(columns=['vomitoxin_ppb'])
y = df['vomitoxin_ppb']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Optimized TPOTRegressor
tpot = TPOTRegressor(
    generations=10,  # Increase evolution time
    population_size=100,  # Increase population
    verbosity=2, 
    random_state=42, 
    n_jobs=-1,
    scoring='neg_mean_absolute_error',  # Focus on reducing MAE
    config_dict='TPOT sparse'  # Restrict to tree-based & sparse models
)

# Fit TPOT to the training data
tpot.fit(X_train, y_train)

# Get predictions
y_pred = tpot.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Model: {tpot.fitted_pipeline_}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Export the best pipeline
tpot.export('best_tpot_pipeline.py')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset (Replace with actual dataset)
# df = pd.read_csv('your_data.csv')
X = df.drop(columns=['vomitoxin_ppb'])
y = df['vomitoxin_ppb']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base model for feature selection
elasticnet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=42)

# Feature Selection using RFE
rfe = RFE(estimator=elasticnet, n_features_to_select=10)

# Normalization
scaler = StandardScaler()

# Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)  # Retain 95% variance

# Define the KNN Regressor
knn = KNeighborsRegressor()

# Create a pipeline
pipeline = Pipeline([
    ('feature_selection', rfe),
    ('scaling', scaler),
    ('pca', pca),
    ('knn', knn)
])

# Define hyperparameter grid for tuning
param_grid = {
    'feature_selection__estimator__l1_ratio': [0.1, 0.35, 0.5, 0.9],
    'knn__n_neighbors': [3, 4, 5, 6],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

# Grid Search for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Model:", best_model)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)


# In[19]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor

# Load dataset
X = df.drop(columns=['vomitoxin_ppb'])
y = df['vomitoxin_ppb']


# Handle NaN values (if any)
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline with Normalization and PCA
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  
    ('stackingestimator', StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.35, tol=1e-05, random_state=42))),
    ('kneighborsregressor', KNeighborsRegressor(n_neighbors=4, p=1, weights='distance'))
])


# Train model
pipeline.fit(X_train, y_train.values.ravel())

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")


# In[13]:


y


# In[20]:


import numpy as np

# Extract feature importance from ElasticNetCV
elastic_net = pipeline.named_steps['stackingestimator'].estimator
importance = np.abs(elastic_net.coef_)

# Rank features by importance
feature_importance = sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)

# Print top features
print("Top Features:")
for feature, score in feature_importance[:20]:  # Show top 20
    print(f"{feature}: {score}")


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(y)
plt.show()


# In[26]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Outlier removal using IQR method
col = "vomitoxin_ppb"

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Split features and target variable
X = df.drop(columns=[col])
y = df[col]

# Handle NaN values (if any)
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline with Normalization and PCA
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  
    ('stackingestimator', StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.35, tol=1e-05, random_state=42))),
    ('kneighborsregressor', KNeighborsRegressor(n_neighbors=4, p=1, weights='distance'))
])

# Train model
pipeline.fit(X_train, y_train.values.ravel())

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")


# In[36]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
#df = pd.read_csv("your_data.csv")  # Replace with actual path

# Remove extreme outliers
col = "vomitoxin_ppb"
df = df[(df[col] > df[col].quantile(0.01)) & (df[col] < df[col].quantile(0.99))]

# Split features and target variable
X = df.drop(columns=[col])
y = df[col]

# Handle NaN values
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train ElasticNet separately
elastic_net = ElasticNetCV(l1_ratio=0.35, tol=1e-05, random_state=42)
elastic_net.fit(X_train, y_train)

# Use ElasticNet predictions as input for RandomForest
y_train_pred = elastic_net.predict(X_train)
y_test_pred = elastic_net.predict(X_test)

# 🔹 Define Pipeline with RandomForest
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  
    ('randomforest', RandomForestRegressor(n_estimators=500, random_state=42))
])

# Train model using ElasticNet predictions
pipeline.fit(X_train, y_train_pred)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test_pred, y_pred)
mse = mean_squared_error(y_test_pred, y_pred)
r2 = r2_score(y_test_pred, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")


# In[43]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
# df = pd.read_csv("your_data.csv")  # Replace with actual dataset path

# Remove extreme outliers
col = "vomitoxin_ppb"
df = df[(df[col] > df[col].quantile(0.01)) & (df[col] < df[col].quantile(0.99))]

# Split features and target variable
X = df.drop(columns=[col])
y = df[col]

# Handle NaN values
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# ✅ Data Augmentation: Add Gaussian noise to X_train
def augment_data(X, y, noise_level=0.01, num_augments=5):
    X_augmented, y_augmented = [], []
    for _ in range(num_augments):
        noise = np.random.normal(0, noise_level, X.shape)
        X_augmented.append(X + noise)
        y_augmented.append(y)
    return np.vstack(X_augmented), np.concatenate(y_augmented)

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Apply Data Augmentation on training set
X_train_aug, y_train_aug = augment_data(X_train.to_numpy(), y_train.to_numpy())

# Convert back to DataFrame
X_train_aug = pd.DataFrame(X_train_aug, columns=X.columns)
y_train_aug = pd.Series(y_train_aug)

# 🔹 Train ElasticNet separately
elastic_net = ElasticNetCV(l1_ratio=0.35, tol=1e-05, random_state=42)
elastic_net.fit(X_train_aug, y_train_aug)

# Use ElasticNet predictions as input for RandomForest
y_train_rf = elastic_net.predict(X_train_aug)
y_test_rf = elastic_net.predict(X_test)

# ✅ Ensuring train-test consistency
X_train_rf, _, y_train_rf, _ = train_test_split(X_train_aug, y_train_rf, test_size=0.2, random_state=42)

# 🔹 Define Pipeline with RandomForest
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  
    ('randomforest', RandomForestRegressor(n_estimators=500, random_state=42))
])

# Train model using ElasticNet predictions
pipeline.fit(X_train_rf, y_train_rf)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test_rf, y_pred)
mse = mean_squared_error(y_test_rf, y_pred)
r2 = r2_score(y_test_rf, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")


# In[ ]:




