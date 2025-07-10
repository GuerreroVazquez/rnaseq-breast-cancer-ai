import pandas as pd
import numpy as np
import pickle
from typing import Literal

#from sklearn.preprocessing import StandardScaler # Import StandardScaler
# ----- Model loading functions -----
estimator_survival_clinical = None
multi_output_model_selected = None
scaler_selected = None
scaler_survival_clinical = None
selected_genes = None
clinincal_variables_needed = ['er status', 'pgr status', 'her2 status',
       'ki67 status', 'nhg']
clinical_features_predicted = ['lymph node group',
       'lymph node status', 'er status', 'pgr status', 'her2 status',
       'ki67 status', 'nhg', 'pam50 subtype']
clinial_features_survival = ['er status', 'pgr status', 'her2 status',
       'ki67 status', 'nhg']

def load_models_and_data():
    """Loads the pickled models and data."""
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
            selected_genes = [line.strip() for line in f]  # Read genes into a list

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: One or more model files not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading models and data: {e}")

def validate_data(dataframe_path:str, data_type: Literal['gene_expression', 'clinical_features']) -> bool:
    """Checks if the input dataframe contains the correct genes or clinical features."""
    global selected_genes
    data = pd.read_csv(dataframe_path)
    if data_type == 'gene_expression':
        if not all(gene in data.columns for gene in selected_genes):
            missing_genes = [gene for gene in selected_genes if gene not in data.columns]
            raise ValueError(f"Error: Gene expression data is missing the following genes: {missing_genes}")
        return True  # Data is valid
    elif data_type == 'clinical_features':
        required_columns = clinincal_variables_needed
        if not all(col in data.columns for col in required_columns):
            missing_columns = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"Error: Clinical data is missing the following columns: {missing_columns}")
        return True  # Data is valid
    else:
        raise ValueError("Error: Invalid data_type. Must be 'gene_expression' or 'clinical_features'")

def predict_clinical_features(gene_expression_data_path:str) -> str:
    """Predicts clinical feature values based on gene expression data."""
    global multi_output_model_selected, scaler_selected, clinical_features_predicted

    try:
        gene_expression_data = pd.read_csv(gene_expression_data_path)
        # Get genes in correct order
        gene_expression_data = gene_expression_data[selected_genes]
        # Scale the gene expression data using the loaded scaler
        scaled_data = scaler_selected.transform(gene_expression_data)
        scaled_data = pd.DataFrame(scaled_data, columns=selected_genes)
        # Predict clinical features
        predicted_clinical_features = multi_output_model_selected.predict(scaled_data)

        # Convert the result to a DataFrame (assuming your model returns a numpy array)
        predicted_df = pd.DataFrame(predicted_clinical_features, columns=clinical_features_predicted)  
        predicted_df.to_csv("user/predicted_features.csv")
        return f"Dataframe with predicted clinical features saved on user/predicted_features.csv"

    except Exception as e:
        raise Exception(f"Error predicting clinical features: {e}")

def predict_survival_outcome(dataframe_path:str, chemo=False, endo=False, days=365) -> list:
    """Predicts survival probabilities for treatment combinations based on clinical features.
    Args:
        clinical_features (pd.DataFrame): DataFrame containing clinical features.
        chemo (bool): Whether to include chemotherapy in the prediction.
        endo (bool): Whether to include endocrine therapy in the prediction.
    Returns:
        list: Predicted survival probabilities for the specified treatment combinations.
    """
    global estimator_survival_clinical, scaler_survival_clinical, clinial_features_survival

    try:
        clinical_features = pd.read_csv(dataframe_path)
        clinical_features = clinical_features[clinial_features_survival]
        clinical_features['endocrine treated'] = endo
        clinical_features['chemo treated'] = chemo
        # Predict survival probabilities
        survival_predictions = estimator_survival_clinical.predict_cumulative_hazard_function(clinical_features)
        survival_probabilities = [np.exp(-chf(days)) for chf in survival_predictions]
        return survival_probabilities

    except Exception as e:
        raise Exception(f"Error predicting survival outcomes: {e}")

def predict_survival_outcomes(dataframe_path:str, days=365) -> str:
    """Predicts survival probabilities for various treatment combinations based on clinical features."""
    global estimator_survival_clinical, scaler_survival_clinical, clinial_features_survival

    try:

        # Predict survival probabilities for all treatment combinations
        survival_probabilities = pd.DataFrame({
            'no treatment': predict_survival_outcome(dataframe_path, chemo=False, endo=False, days=days),
            'endocrine treatment': predict_survival_outcome(dataframe_path, chemo=False, endo=True, days=days),
            'chemotherapy': predict_survival_outcome(dataframe_path, chemo=True, endo=False, days=days),
            'chemotherapy + endocrine treatment': predict_survival_outcome(dataframe_path, chemo=True, endo=True, days=days)
        })
        survival_probabilities.to_csv("user/predicted_survivial_probabilities.csv")
        return f"Dataframe with predicted clinical features saved on user/predicted_survivial_probabilities.csv"


    except Exception as e:
        raise Exception(f"Error predicting survival outcomes: {e}")

def recommend_treatment(clinical_features_path: str) -> str:
    """Recommends the best treatment option based on survival probabilities."""

    try:
        survival_probabilities = predict_survival_outcomes(clinical_features_path)
        # Save the results to a dictionary
        survival_probabilities = survival_probabilities.to_dict()
        return  survival_probabilities

    except Exception as e:
        raise Exception(f"Error recommending treatment: {e}")

def get_column_names(data_path: str) -> list:
    """Returns a list of first 10 column names in the dataframe."""
    try:
        data = pd.read_csv(data_path)
        return list(data.columns)[:10]
    except Exception as e:
        raise Exception(f"Error getting column names: {e}")

def rename_columns(dataframe_path:str, column_mapping: list[str]) -> str:
    """Renames columns in the dataframe based on the provided mapping."""
    try:
        data = read_data_from_csv(dataframe_path)
        data.rename(columns=column_mapping)
        data.to_csv(dataframe_path)
        return f"Data columns renamed, saved on {dataframe_path}"
    except Exception as e:
        raise Exception(f"Error renaming columns: {e}")

def read_data_from_csv(file_path: str) -> pd.DataFrame:
    """Reads data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error reading data from CSV: {e}")
    
load_models_and_data()