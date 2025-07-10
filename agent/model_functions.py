import pandas as pd
import numpy as np
import pickle
from typing import Literal

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

def validate_data(dataframe_path: str, data_type: Literal['gene_expression', 'clinical_features']) -> bool:
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
    data = pd.read_csv(dataframe_path)

    if data_type == 'gene_expression':
        missing_genes = [gene for gene in selected_genes if gene not in data.columns]
        if missing_genes:
            raise ValueError(f"Missing genes: {missing_genes}")
    elif data_type == 'clinical_features':
        missing = [col for col in clinincal_variables_needed if col not in data.columns]
        if missing:
            raise ValueError(f"Missing clinical features: {missing}")
    else:
        raise ValueError("data_type must be 'gene_expression' or 'clinical_features'.")

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

    try:
        gene_expression_data = pd.read_csv(gene_expression_data_path)[selected_genes]
        scaled_data = scaler_selected.transform(gene_expression_data)
        predictions = multi_output_model_selected.predict(scaled_data)
        predicted_df = pd.DataFrame(predictions, columns=clinical_features_predicted)
        output_path = "user/predicted_features.csv"
        predicted_df.to_csv(output_path, index=False)
        return output_path

    except Exception as e:
        raise Exception(f"Prediction failed: {e}")

def predict_survival_outcome(dataframe_path: str, chemo: bool = False, endo: bool = False, days: int = 365) -> list:
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

    try:
        clinical_features = pd.read_csv(dataframe_path)[clinical_features_survival]
        clinical_features['endocrine treated'] = endo
        clinical_features['chemo treated'] = chemo

        chf_list = estimator_survival_clinical.predict_cumulative_hazard_function(clinical_features)
        return [float(np.exp(-chf(days))) for chf in chf_list]

    except Exception as e:
        raise Exception(f"Survival prediction failed: {e}")

def predict_survival_outcomes(dataframe_path: str, days: int = 365) -> dict:
    """
    Predicts survival probabilities for four treatment combinations.

    Args:
        dataframe_path (str): Path to CSV with clinical features.
        days (int): Time in days to compute survival probabilities.

    Returns:
        dict: Dictionary of the survival probabilities.
    """
    try:
        df = pd.DataFrame({
            'no treatment': predict_survival_outcome(dataframe_path, False, False, days),
            'endocrine treatment': predict_survival_outcome(dataframe_path, False, True, days),
            'chemotherapy': predict_survival_outcome(dataframe_path, True, False, days),
            'chemotherapy + endocrine': predict_survival_outcome(dataframe_path, True, True, days)
        })
        return df.to_dict()


    except Exception as e:
        raise Exception(f"Survival outcomes prediction failed: {e}")

def get_column_names(data_path: str) -> list:
    """
    Returns first 10 column names from a CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        list: List of column names.
    """
    try:
        return list(pd.read_csv(data_path).columns[:10])
    except Exception as e:
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
    try:
        df = pd.read_csv(dataframe_path)
        df = df.rename(columns=column_mapping)
        df.to_csv(dataframe_path, index=False)
        return dataframe_path
    except Exception as e:
        raise Exception(f"Renaming columns failed: {e}")

def read_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file as a pandas DataFrame.

    Args:
        file_path (str): Path to the file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

# Load models on startup
load_models_and_data()
