# Task 7: Support Vector Machines (SVM)
# Objective: Linear and Non-linear Classification with SVM

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load dataset (Breast Cancer)
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train SVM with linear kernel
svm_linear = SVC(kernel="linear", C=1)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

print("\n🔹 Linear Kernel SVM Results")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print(confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# 3. Train SVM with RBF kernel
svm_rbf = SVC(kernel="rbf", C=1, gamma=0.1)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

print("\n🔹 RBF Kernel SVM Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print(confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# 4. Cross-validation
scores = cross_val_score(SVC(kernel="rbf", C=1, gamma=0.1), X, y, cv=5)
print("\n🔹 Cross-validation Accuracy Scores:", scores)
print("Mean CV Accuracy:", scores.mean())

# 5. Visualization (only for 2D data demo)
# We'll use a simple dataset (make_moons) to show decision boundaries
from sklearn.datasets import make_moons

X_vis, y_vis = make_moons(n_samples=200, noise=0.2, random_state=42)
X_vis = StandardScaler().fit_transform(X_vis)

svm_vis = SVC(kernel="rbf", C=1, gamma=0.5)
svm_vis.fit(X_vis, y_vis)

# Create meshgrid
xx, yy = np.meshgrid(
    np.linspace(X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1, 500),
    np.linspace(X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1, 500)
)
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, s=30, cmap=plt.cm.coolwarm, edgecolors="k")
plt.title("SVM with RBF Kernel (Decision Boundary)")
plt.show()
