# Task 3: Linear Regression
# Objective: Implement and understand simple & multiple linear regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Step 1: Import and Preprocess Dataset
# -----------------------------

# Example: House Price dataset (you can replace with your own CSV)
# Let's create a small dataset for demonstration (if no CSV available)
data = {
    'Area': [1500, 1600, 1700, 1800, 2000, 2300, 2500, 2700, 3000, 3200],
    'Bedrooms': [3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    'Price': [330000, 340000, 360000, 400000, 430000, 480000, 500000, 550000, 620000, 680000]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# Features (X) and Target (y)
X = df[['Area', 'Bedrooms']]   # multiple features
y = df['Price']

# -----------------------------
# Step 2: Split into Train-Test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 3: Fit Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# -----------------------------
# Step 5: Interpret Coefficients
# -----------------------------
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {list(zip(X.columns, model.coef_))}")

# -----------------------------
# Step 6: Simple Linear Regression Plot (Area vs Price)
# -----------------------------
plt.scatter(df['Area'], df['Price'], color='blue', label="Actual Prices")
plt.plot(df['Area'], model.predict(df[['Area', 'Bedrooms']]), color='red', label="Regression Line")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("Simple Linear Regression (Area vs Price)")
plt.legend()
plt.show()
