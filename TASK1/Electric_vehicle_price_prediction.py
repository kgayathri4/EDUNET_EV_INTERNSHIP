# =============================
# Electric Vehicle Price Prediction
# =============================

# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load dataset
df = pd.read_csv('EV_cars.csv')

print("Rows, Cols: ", df.shape)
print(df.head())

# Step 3: Check info and missing values
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

print("\nColumn names: ", df.columns.tolist())

# Step 4: Clean data
df = df.rename(columns={
    'Price.DE.': 'Price',
    'acceleration..0.100.': 'Acceleration_0_100'
})

# Drop rows with missing target (Price)
df = df.dropna(subset=['Price'])

# Fill missing Fast_charge values with median
df['Fast_charge'] = df['Fast_charge'].fillna(df['Fast_charge'].median())

# Drop unnecessary columns
df = df.drop(['Car_name_link', 'Car_name'], axis=1)

print("\n After Cleaning:")
print("Rows, Cols:", df.shape)
print(df.columns.tolist())

# Step 5: Define features and target
feature_cols = ['Battery', 'Efficiency', 'Fast_charge', 'Range', 'Top_speed', 'Acceleration_0_100']
target_col = 'Price'

X = df[feature_cols]
y = df[target_col]

print("\nFeatures used:", feature_cols)

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

# Step 7: Preprocessing
# Separate numerical and categorical columns (all numeric here, but kept flexible)
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

num_pipe = Pipeline([('scaler', StandardScaler())])
cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# Step 8: Model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Step 9: Train model
model.fit(X_train, y_train)

# Step 10: Predict
y_pred = model.predict(X_test)

# Step 11: Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 3))

# Step 12: Example Prediction
sample = X_test.iloc[0:1]
pred_price = model.predict(sample)
print("\nExample Prediction:")
print("Input Data:\n", sample)
print("Predicted Price (€):", round(pred_price[0], 2))

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("Actual vs Predicted EV Prices")
plt.show()

import joblib
joblib.dump(model, 'ev_price_pipeline.pkl')
print("\n Model saved as ev_price_pipeline.pkl")