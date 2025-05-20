import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_FILE_PATH = os.path.join("data", "collected_hand_poses_metadata.csv")
TARGET_COLUMN = 'label'
FEATURE_COLUMNS = [f'{axis}{i}_rel' for i in range(21) for axis in ('x', 'y', 'z')]

if not os.path.exists(CSV_FILE_PATH):
    print(f"Error: CSV file not found at path: {CSV_FILE_PATH}")
    print("Please check the path and file name.")
    exit()

try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Loaded {len(df)} rows of data from {CSV_FILE_PATH}")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].replace('', np.nan)
        X[col] = X[col].replace(' ', np.nan)

X = X.astype(float)

print(f"\nNumber of missing values per feature BEFORE imputation:\n{X.isnull().sum()[X.isnull().sum() > 0]}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nClasses (labels) and their encoded values: {list(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

imputer = SimpleImputer(strategy='mean')
X_train_imputed_rf = imputer.fit_transform(X_train)
X_test_imputed_rf = imputer.transform(X_test)

X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()

print(f"\nNumber of missing values in X_train_xgb after ensuring NaN:\n{X_train_xgb.isnull().sum()[X_train_xgb.isnull().sum() > 0]}")

print("\n--- Training RandomForest model ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_imputed_rf, y_train)
y_pred_rf = rf_model.predict(X_test_imputed_rf)

print("\n--- Training XGBoost model ---")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_xgb, y_train)
y_pred_xgb = xgb_model.predict(X_test_xgb)

print("\n\n--- Model Evaluation ---")

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("\nRandomForest Classifier:")
print(f"Accuracy: {accuracy_rf:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_, zero_division=0))
cm_rf = confusion_matrix(y_test, y_pred_rf)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("\nXGBoost Classifier:")
print(f"Accuracy: {accuracy_xgb:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_, zero_division=0))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(filename)
    print(f"Confusion matrix saved as: {filename}")
    plt.close()

plot_confusion_matrix(cm_rf, label_encoder.classes_, "Confusion Matrix (RandomForest)")
plot_confusion_matrix(cm_xgb, label_encoder.classes_, "Confusion Matrix (XGBoost)")

print("\n--- Model Comparison ---")
print(f"RandomForest Accuracy: {accuracy_rf:.4f}")
print(f"XGBoost Accuracy:    {accuracy_xgb:.4f}")