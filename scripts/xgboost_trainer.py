import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(project_root_dir, relative_csv_path="data/collected_hand_poses.csv"):
    csv_full_path = os.path.join(project_root_dir, relative_csv_path)
    print(f"Attempting to load data from: {csv_full_path}")
    try:
        df = pd.read_csv(csv_full_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {csv_full_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None

def preprocess_data_xgb(df):
    if df is None:
        return None, None, None, None

    df_processed = df.copy()
    df_processed.dropna(subset=['label'], inplace=True)
    if df_processed.empty:
        print("Error: No data left after dropping rows with missing labels.")
        return None, None, None, None

    feature_columns = []
    for i in range(21):
        feature_columns.extend([f'x{i}_rel', f'y{i}_rel', f'z{i}_rel'])
    
    existing_feature_columns = [col for col in feature_columns if col in df_processed.columns]
    
    if not existing_feature_columns:
        print("Error: No expected feature columns found in the DataFrame.")
        return None, None, None, None
        
    X = df_processed[existing_feature_columns].copy()
    y_text = df_processed['label'].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_text)
    
    print(f"Number of features (columns) used for X: {len(X.columns)}")
    if X.isnull().sum().sum() > 0:
        print(f"Info: Feature data contains {X.isnull().sum().sum()} NaN values. XGBoost will handle them.")
    
    return X, y_encoded, label_encoder, existing_feature_columns

def save_results(model, label_encoder, y_true_test, y_pred_test, model_name_prefix, output_subdir):
    """Saves the model, label encoder, classification report, and confusion matrix."""
    os.makedirs(output_subdir, exist_ok=True)
    
    model_path = os.path.join(output_subdir, f"{model_name_prefix}_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    encoder_path = os.path.join(output_subdir, f"label_encoder_{model_name_prefix}.joblib")
    joblib.dump(label_encoder, encoder_path)
    print(f"LabelEncoder saved to: {encoder_path}")

    report_str = classification_report(y_true_test, y_pred_test, target_names=label_encoder.classes_, zero_division=0)
    accuracy = accuracy_score(y_true_test, y_pred_test)
    
    report_path = os.path.join(output_subdir, f"{model_name_prefix}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Accuracy on test set: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    print(f"Classification report saved to: {report_path}")

    cm = confusion_matrix(y_true_test, y_pred_test, labels=np.arange(len(label_encoder.classes_)))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name_prefix}')
    cm_path = os.path.join(output_subdir, f"{model_name_prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")

def tune_and_train_xgboost(X_train, y_train, X_test, y_test, feature_names, label_encoder, models_output_dir):
    print("\n--- Training and Tuning XGBoost ---")
    
    output_subdir_xgb = os.path.join(models_output_dir, "xgboost")

    xgb_clf = xgb.XGBClassifier(objective='multi:softprob', 
                                eval_metric='mlogloss', 
                                use_label_encoder=False, 
                                random_state=42)

    param_grid_xgb = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2] 
    }

    # param_grid_xgb_small = {
    #     'n_estimators': [100, 200], 'max_depth': [3, 5],
    #     'learning_rate': [0.1, 0.2], 'subsample': [0.8], 'colsample_bytree': [0.8]
    # }

    grid_search_xgb = GridSearchCV(estimator=xgb_clf, 
                                   param_grid=param_grid_xgb,
                                   scoring='accuracy', 
                                   cv=3, 
                                   verbose=2,
                                   n_jobs=-1) 

    print("Starting GridSearchCV for XGBoost...")
    grid_search_xgb.fit(X_train, y_train)

    print("\nXGBoost Hyperparameter tuning finished.")
    print(f"Best XGBoost parameters: {grid_search_xgb.best_params_}")
    print(f"Best XGBoost cross-validation accuracy: {grid_search_xgb.best_score_:.4f}")
    
    best_model = grid_search_xgb.best_estimator_

    y_pred_test = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"XGBoost Test Set Accuracy with best model: {test_accuracy:.4f}")

    save_results(best_model, label_encoder, y_test, y_pred_test, "xgboost", output_subdir_xgb)
    
    return best_model, test_accuracy

def main():
    project_root = get_project_root()
    models_output_dir = os.path.join(project_root, "models")

    print("--- XGBoost Training Pipeline Started ---")
    df = load_data(project_root_dir=project_root)
    if df is None:
        return

    X, y_encoded, label_encoder, feature_names = preprocess_data_xgb(df)
    if X is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    print(f"\nData split into training and testing sets:")
    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    if label_encoder:
        print(f"Classes: {list(label_encoder.classes_)}")

    tune_and_train_xgboost(X_train, y_train, X_test, y_test, feature_names, label_encoder, models_output_dir)
    
    print("\n--- XGBoost Training Pipeline Finished ---")

if __name__ == "__main__":
    main()