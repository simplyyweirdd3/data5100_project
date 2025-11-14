import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# 1. Load data
df = pd.read_csv("diabetes_prediction_dataset.csv")

# 2. Detect target column
possible_targets = ["diabetes", "Diabetes", "Outcome", "outcome"]
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError("Could not find target column. Check df.columns().")

print(f"Using target column: {target_col}")

# 3. Encode categorical columns using one hot encoding
df_encoded = df.copy()
cat_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()

if cat_cols:
    df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

# 4. Split features and target
X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 6a. Correlation heatmap (using encoded data)
plt.figure(figsize=(10, 8))
corr = df_encoded.corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Encoded Features and Target")
plt.tight_layout()
plt.show()

# 6b. Outcome distribution plot
plt.figure(figsize=(4, 4))
sns.countplot(x=y)
plt.title("Outcome Distribution")
plt.xlabel(target_col)
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 6c. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest on Diabetes Dataset")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
