
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# 1. Import dataset
df = sns.load_dataset("titanic")

print("Original Data (first 5 rows):")
print(df.head())

# Explore dataset
print("\nBasic Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe(include="all"))

# 2. Handle Missing Values
# Fill age with median
df["age"].fillna(df["age"].median(), inplace=True)

# Fill embarked with mode (most common value)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# Drop columns with too many missing values (like 'deck')
df.drop(columns=["deck"], inplace=True)

# 3. Convert Categorical → Numerical
# Encode sex (Male=1, Female=0)
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])

# One-hot encoding for "class" and "embarked"
df = pd.get_dummies(df, columns=["class", "embarked"], drop_first=True)

# 4. Normalize/Standardize
# Example with age & fare
scaler = StandardScaler()
df[["age", "fare"]] = scaler.fit_transform(df[["age", "fare"]])

# 5. Detect Outliers using Boxplot & Remove them (on fare)
plt.figure(figsize=(6,4))
sns.boxplot(x=df["fare"])
plt.title("Boxplot of Fare (Before Outlier Removal)")
plt.show()

# IQR Method
Q1 = df["fare"].quantile(0.25)
Q3 = df["fare"].quantile(0.75)
IQR = Q3 - Q1

df = df[(df["fare"] >= Q1 - 1.5*IQR) & (df["fare"] <= Q3 + 1.5*IQR)]

plt.figure(figsize=(6,4))
sns.boxplot(x=df["fare"])
plt.title("Boxplot of Fare (After Outlier Removal)")
plt.show()

# Final Cleaned Data
print("\nFinal Cleaned Data (first 5 rows):")
print(df.head())
print("\nShape of dataset after cleaning:", df.shape)
