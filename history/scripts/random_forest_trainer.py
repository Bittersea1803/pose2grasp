import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
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

def preprocess_data_rf(df):
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
        print("Error: No expected feature columns found.")
        return None, None, None, None
        
    X = df_processed[existing_feature_columns].copy()
    y_text = df_processed['label'].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_text)
    
    print(f"Number of features (columns) used for X: {len(X.columns)}")
    
    if X.isnull().sum().sum() > 0:
        print(f"Info: Raw feature data contains {X.isnull().sum().sum()} NaN values. Imputation will be handled after train/test split.")
    else:
        print("Info: No NaN values found in raw features.")
        
    return X, y_encoded, label_encoder, existing_feature_columns

def save_results(model, label_encoder, imputer, y_true_test, y_pred_test, model_name_prefix, output_subdir):
    os.makedirs(output_subdir, exist_ok=True)
    
    model_path = os.path.join(output_subdir, f"{model_name_prefix}_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    encoder_path = os.path.join(output_subdir, f"label_encoder_{model_name_prefix}.joblib")
    joblib.dump(label_encoder, encoder_path)
    print(f"LabelEncoder saved to: {encoder_path}")

    if imputer:
        imputer_path = os.path.join(output_subdir, f"imputer_{model_name_prefix}.joblib")
        joblib.dump(imputer, imputer_path)
        print(f"Imputer saved to: {imputer_path}")

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


def tune_and_train_random_forest(X_train, y_train, X_test, y_test, feature_names, label_encoder, imputer, models_output_dir):
    print("\n--- Training and Tuning Random Forest ---")
    output_subdir_rf = os.path.join(models_output_dir, "random_forest")

    rf_clf = RandomForestClassifier(random_state=42)

    param_grid_rf = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None] 
    }
    # param_grid_rf_small = {
    #     'n_estimators': [100, 150], 'max_depth': [10, 20],
    #     'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]
    # }


    grid_search_rf = GridSearchCV(estimator=rf_clf, 
                                  param_grid=param_grid_rf,
                                  scoring='accuracy', 
                                  cv=3, 
                                  verbose=2,
                                  n_jobs=-1) 

    print("Starting GridSearchCV for Random Forest...")
    grid_search_rf.fit(X_train, y_train) 

    print("\nRandom Forest Hyperparameter tuning finished.")
    print(f"Best Random Forest parameters: {grid_search_rf.best_params_}")
    print(f"Best Random Forest cross-validation accuracy: {grid_search_rf.best_score_:.4f}")
    
    best_model = grid_search_rf.best_estimator_
    
    # X_test is already imputed when passed to this function
    y_pred_test = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"Random Forest Test Set Accuracy with best model: {test_accuracy:.4f}")

    save_results(best_model, label_encoder, imputer, y_test, y_pred_test, "random_forest", output_subdir_rf)
    
    return best_model, test_accuracy

def main():
    project_root = get_project_root()
    models_output_dir = os.path.join(project_root, "models")

    print("--- Random Forest Training Pipeline Started ---")
    df = load_data(project_root_dir=project_root)
    if df is None:
        return

    X_raw, y_encoded, label_encoder, feature_names = preprocess_data_rf(df) # Expects 4 values
    if X_raw is None:
        return

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=feature_names)
    X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=feature_names)

    print(f"\nData split and imputed:")
    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    if label_encoder:
        print(f"Classes: {list(label_encoder.classes_)}")

    tune_and_train_random_forest(X_train, y_train, X_test, y_test, feature_names, label_encoder, imputer, models_output_dir)
    
    print("\n--- Random Forest Training Pipeline Finished ---")

if __name__ == "__main__":
    main()