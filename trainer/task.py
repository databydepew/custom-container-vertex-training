# Import libraries
import os
import joblib
import logging
import argparse
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

# Input Arguments
parser.add_argument(
    '--dataset_dir',
    help='Dataset file on Google Cloud Storage',
    type=str,
    required=True
)

parser.add_argument(
    '--model_dir',
    help='Directory to output model artifacts',
    type=str,
    default=os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else ""
)

parser.add_argument(
    '--hypertune',
    help='Whether to run hyperparameter tuning',
    type=str,
    default="False"
)

parser.add_argument(
    '--learning_rate',
    help='Learning rate for XGBoost',
    type=float,
    default=0.1
)

parser.add_argument(
    '--max_depth',
    help='Maximum depth of trees',
    type=int,
    default=6
)

parser.add_argument(
    '--n_estimators',
    help='Number of estimators',
    type=int,
    default=100
)

# Parse arguments
args = parser.parse_args()
arguments = vars(args)

try:
    # Load dataset
    data_gcs_path = arguments['dataset_dir']
    logging.info(f"Reading dataset from: {data_gcs_path}")
    df = pd.read_csv(data_gcs_path)

    # Separate features and labels
    X, y = df.drop(columns=['refinance']), df['refinance'].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0)

    # Train model with hyperparameters
    model = xgb.XGBClassifier(
        learning_rate=arguments['learning_rate'],
        max_depth=arguments['max_depth'],
        n_estimators=arguments['n_estimators']
    ).fit(X_train, y_train)
    
    # Evaluate the model
    y_hat = model.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])

    # Log metrics for Vertex AI hyperparameter tuning
    print(f"metric:accuracy={acc}")
    print(f"metric:roc_auc_score={auc}")
    logging.info(f"Accuracy: {acc}")
    logging.info(f"AUC: {auc}")

    # Save model artifact
    artifact_filename = 'model.joblib'
    joblib.dump(model, artifact_filename)
    logging.info(f"Model saved locally as: {artifact_filename}")

    # Upload model artifact to GCS
    model_directory = arguments['model_dir']
    if model_directory:
        storage_path = os.path.join(model_directory, artifact_filename)
        blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
        blob.upload_from_filename(artifact_filename)
        logging.info(f"Model uploaded to: {storage_path}")
    else:
        logging.info("Model directory not specified. Skipping upload to GCS.")

except Exception as e:
    logging.error(f"An error occurred during training: {e}")
    raise