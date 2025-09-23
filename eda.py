# =========================
# Exploratory Data Analysis (EDA) on Titanic Dataset
# =========================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Titanic Dataset from web
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 3. View dataset
print("First 5 rows:")
print(df.head(), "\n")

# 4. Dataset Info
print("Dataset Info:")
print(df.info(), "\n")

# 5. Summary Statistics
print("Summary Statistics:")
print(df.describe(include='all'), "\n")

# 6. Missing Values
print("Missing Values:")
print(df.isnull().sum(), "\n")

# 7. Histogram: Age
plt.figure(figsize=(6,4))
sns.histplot(df['Age'].dropna(), kde=True, bins=30, color='skyblue')
plt.title("Age Distribution")
plt.show()
# Observation:
# - Most passengers are between 20-40 years old.
# - Some ages are missing (NaN), which may need handling.

# 8. Histogram: Fare
plt.figure(figsize=(6,4))
sns.histplot(df['Fare'].dropna(), kde=True, bins=30, color='orange')
plt.title("Fare Distribution")
plt.show()
# Observation:
# - Most passengers paid lower fares.
# - Few passengers paid very high fares (outliers).

# 9. Boxplot: Age vs Survival
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()
# Observation:
# - Younger passengers (children) had higher chances of survival.
# - Older passengers had lower survival chances.

# 10. Boxplot: Fare vs Pclass
plt.figure(figsize=(6,4))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare by Passenger Class")
plt.show()
# Observation:
# - 1st class passengers paid higher fares than 2nd and 3rd class.
# - 3rd class fares are mostly low.

# 11. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
# Observation:
# - 'Fare' is slightly positively correlated with 'Survived'.
# - 'Pclass' is negatively correlated with 'Survived' (higher class, better survival).

# 12. Pairplot: Age, Fare, Pclass, Survived
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']].dropna(), hue='Survived')
plt.show()
# Observation:
# - Survivors are generally younger and from higher classes.
# - Higher fare passengers tend to survive more.

# =========================
# Overall Observations:
# - Children and younger passengers had better survival chances.
# - Females survived more than males (from earlier analysis of Sex feature).
# - 1st class passengers survived more compared to 2nd and 3rd class.
# - Wealthier passengers (higher fare) had slightly better survival.
# - Dataset has missing values in Age and Cabin columns, which may need preprocessing for modeling.
# =========================
