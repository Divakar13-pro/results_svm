import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Decision Tree
# ----------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

# Save Decision Tree visualization
plt.figure(figsize=(15, 10))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=["0", "1"])
plt.savefig("decision_tree.png")
plt.close()

# ----------------------------
# Random Forest
# ----------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# Save Random Forest Feature Importances
importances = rf.feature_importances_
plt.figure(figsize=(12, 6))
plt.barh(X.columns, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importances")
plt.savefig("random_forest_importances.png")
plt.close()

# ----------------------------
# Evaluation (store in results.txt)
# ----------------------------
with open("results.txt", "w") as f:
    f.write("Decision Tree Performance:\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_dt)}\n")
    f.write(classification_report(y_test, y_pred_dt))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_dt)))
    f.write("\n\n")

    f.write("Random Forest Performance:\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_rf)))
