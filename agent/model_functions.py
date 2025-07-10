import logging
# log into model_functions
import pandas as pd
import numpy as np
import pickle
from typing import Literal


logger = logging.getLogger(__name__)  # Or use '' for root logger
logger.setLevel(logging.INFO)

# Add file handler directly
file_handler = logging.FileHandler('SCANB_AI_model_functions.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

logger.info("This WILL go to the file!")

# Global variables for loaded models and configuration
estimator_survival_clinical = None
multi_output_model_selected = None
scaler_selected = None
scaler_survival_clinical = None
selected_genes = None

# Required columns
clinincal_variables_needed = ['er status', 'pgr status', 'her2 status', 'ki67 status', 'nhg']
clinical_features_predicted = ['lymph node group', 'lymph node status', 'er status', 'pgr status', 'her2 status', 'ki67 status', 'nhg', 'pam50 subtype']
clinical_features_survival = clinincal_variables_needed  # rename for clarity

# ---------------------------- #
#        Core Functions        #
# ---------------------------- #

def load_models_and_data():
    """
    Loads pre-trained models and preprocessing objects from disk into global variables.

    Raises:
        FileNotFoundError: If any model file is missing.
        Exception: If any error occurs during loading.
    """
    global estimator_survival_clinical, multi_output_model_selected, scaler_selected, scaler_survival_clinical, selected_genes
    logging.info("Loading models and data...")
    try:
        with open('models/estimator_survival_clinical.pkl', 'rb') as f:
            estimator_survival_clinical = pickle.load(f)
        with open('models/scaler_survival_clinical.pkl', 'rb') as f:
            scaler_survival_clinical = pickle.load(f)
        with open('models/multi_output_model_selected_.pkl', 'rb') as f:
            multi_output_model_selected = pickle.load(f)
        with open('models/scaler_selected.pkl', 'rb') as f:
            scaler_selected = pickle.load(f)
        with open('models/selected_genes.txt', 'r') as f:
            selected_genes = [line.strip() for line in f]

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing model file: {e}")
    except Exception as e:
        raise Exception(f"Model loading failed: {e}")

def validate_data(dataframe_path: str, data_type: str) -> bool:
    """
    Validates that a CSV file has all the necessary columns for the specified data type.

    Args:
        dataframe_path (str): Path to CSV file.
        data_type (Literal): Either 'gene_expression' or 'clinical_features'.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If required columns are missing.
    """
    global selected_genes
    logging.info(f"Validating data from {dataframe_path} for data type {data_type}...")
    data = pd.read_csv(dataframe_path)
    if data_type not in ['gene_expression', 'clinical_features']:
        raise ValueError("data_type must be 'gene_expression' or 'clinical_features'")

    if data_type == 'gene_expression':
        missing_genes = [gene for gene in selected_genes if gene not in data.columns]
        if missing_genes:
            raise ValueError(f"Missing genes: {missing_genes}")
    elif data_type == 'clinical_features':
        missing = [col for col in clinincal_variables_needed if col not in data.columns]
        if missing:
            logging.error(f"Missing clinical features: {missing}")
            raise ValueError(f"Missing clinical features: {missing}")
    else:
        logging.error("Invalid data type provided for validation.")
        raise ValueError("data_type must be 'gene_expression' or 'clinical_features'.")
    logging.info("Data validation passed.")
    return True

def predict_clinical_features(gene_expression_data_path: str) -> str:
    """
    Predicts clinical features from gene expression data.

    Args:
        gene_expression_data_path (str): Path to CSV containing gene expression data.

    Returns:
        str: Path to CSV with predicted clinical features.
    """
    global multi_output_model_selected, scaler_selected, selected_genes
    logging.info(f"Predicting clinical features from {gene_expression_data_path}...")
    try:
        gene_expression_data = pd.read_csv(gene_expression_data_path)[selected_genes]
        scaled_data = scaler_selected.transform(gene_expression_data)
        predictions = multi_output_model_selected.predict(scaled_data)
        predicted_df = pd.DataFrame(predictions, columns=clinical_features_predicted)
        output_path = "user/predicted_features.csv"
        predicted_df.to_csv(output_path, index=False)
        logging.info(f"Clinical features predicted and saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise Exception(f"Prediction failed: {e}")

def predict_survival_outcome(dataframe_path: str, chemo: bool, endo: bool, days: int) -> list:
    """
    Predicts survival probabilities given a treatment scenario.

    Args:
        dataframe_path (str): Path to CSV with clinical features.
        chemo (bool): Use chemotherapy.
        endo (bool): Use endocrine therapy.
        days (int): Time in days to compute survival probability.

    Returns:
        list: Survival probabilities per sample.
    """
    global estimator_survival_clinical, scaler_survival_clinical
    if days is None:
        days = 365
    logging.info(f"Predicting survival outcome from {dataframe_path} with chemo={chemo}, endo={endo}, days={days}...")
    try:
        clinical_features = pd.read_csv(dataframe_path)[clinical_features_survival]
        clinical_features['endocrine treated'] = endo
        clinical_features['chemo treated'] = chemo

        chf_list = estimator_survival_clinical.predict_cumulative_hazard_function(clinical_features)
        survival = [float(np.exp(-chf(days))) for chf in chf_list]
        logging.info(f"Cumulative hazard function computed successfully. Survival probabilities: {survival}")
        return survival

    except Exception as e:
        logging.error(f"Survival prediction failed: {e}")
        raise Exception(f"Survival prediction failed: {e}")

def predict_survival_outcomes(dataframe_path: str, days: int) -> dict:
    """
    Predicts survival probabilities for four treatment combinations.

    Args:
        dataframe_path (str): Path to CSV with clinical features.
        days (int): Time in days to compute survival probabilities.

    Returns:
        dict: Dictionary of the survival probabilities.
    """
    if days is None:
        days = 365
    logging.info(f"Predicting survival outcomes from {dataframe_path} for {days} days...")
    try:
        df = pd.DataFrame({
            'no treatment': predict_survival_outcome(dataframe_path, False, False, days),
            'endocrine treatment': predict_survival_outcome(dataframe_path, False, True, days),
            'chemotherapy': predict_survival_outcome(dataframe_path, True, False, days),
            'chemotherapy + endocrine': predict_survival_outcome(dataframe_path, True, True, days)
        })
        df_dict = df.to_dict()
        logging.info(f"Survival outcomes prediction completed successfully. Results: {df_dict}")
        return df_dict


    except Exception as e:
        logging.error(f"Survival outcomes prediction failed: {e}")
        raise Exception(f"Survival outcomes prediction failed: {e}")

def get_column_names(data_path: str) -> list:
    """
    Returns first 10 column names from a CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        list: List of column names.
    """
    logging.info(f"Reading column names from {data_path}...")
    try:
        col_names = list(pd.read_csv(data_path).columns[:10])
        logging.info(f"Column names read successfully: {col_names}")
        return col_names
    except Exception as e:
        logging.error(f"Reading columns failed: {e}")
        raise Exception(f"Reading columns failed: {e}")

def rename_columns(dataframe_path: str, column_mapping: dict) -> str:
    """
    Renames columns in a CSV file based on mapping.

    Args:
        dataframe_path (str): Path to CSV file.
        column_mapping (dict): Dictionary with old names as keys and new names as values.

    Returns:
        str: Path to updated CSV file.
    """
    logging.info(f"Renaming columns in {dataframe_path} with mapping {column_mapping}...")
    try:
        df = pd.read_csv(dataframe_path)
        df = df.rename(columns=column_mapping)
        df.to_csv(dataframe_path, index=False)
        logging.info(f"Columns renamed successfully in {dataframe_path}.")
         # Log the renamed columns
        logging.info(f"Renamed columns: {list(df.columns)}")
        return dataframe_path
    except Exception as e:
        logging.error(f"Renaming columns failed: {e}")
        raise Exception(f"Renaming columns failed: {e}")

def read_data_from_csv(file_path: str) -> list[dict]:
    """
    Loads a CSV file as a pandas DataFrame.

    Args:
        file_path (str): Path to the file.

    Returns:
        list of dict: The loaded data.
    """
    try:
        logging.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}.")
        return df.to_dict(orient='records')  # Convert DataFrame to list of dicts
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise Exception(f"Failed to load data: {e}")

# Load models on startup
load_models_and_data()
