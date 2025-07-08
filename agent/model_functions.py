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
clinincal_variables_needed = ['age at diagnosis', 'tumor size', 'lymph node group',
       'lymph node status', 'er status', 'pgr status', 'her2 status',
       'ki67 status', 'nhg', 'pam50 subtype']

def load_models_and_data():
    """Loads the pickled models and data."""
    global estimator_survival_clinical, multi_output_model_selected, scaler_selected, scaler_survival_clinical, selected_genes

    try:
        with open('estimator_survival_clinical.pkl', 'rb') as f:
            estimator_survival_clinical = pickle.load(f)
        with open('scaler_survival_clinical.pkl', 'rb') as f:
            scaler_survival_clinical = pickle.load(f)
        with open('multi_output_model_selected.pkl', 'rb') as f:
            multi_output_model_selected = pickle.load(f)
        with open('scaler_selected.pkl', 'rb') as f:
            scaler_selected = pickle.load(f)
        with open('selected_genes.txt', 'r') as f:
            selected_genes = [line.strip() for line in f]  # Read genes into a list

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: One or more model files not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading models and data: {e}")

def validate_data(data:pd.DataFrame, data_type: Literal['gene_expression', 'clinical_features']) -> bool:
    """Checks if the input dataframe contains the correct genes or clinical features."""
    global selected_genes

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

def predict_clinical_features(gene_expression_data: pd.DataFrame) -> pd.DataFrame:
    """Predicts clinical feature values based on gene expression data."""
    global multi_output_model_selected, scaler_selected

    try:
        # Scale the gene expression data using the loaded scaler
        scaled_data = scaler_selected.transform(gene_expression_data)

        # Predict clinical features
        predicted_clinical_features = multi_output_model_selected.predict(scaled_data)

        # Convert the result to a DataFrame (assuming your model returns a numpy array)
        predicted_df = pd.DataFrame(predicted_clinical_features, columns=['age', 'er', 'pgr', 'her2', 'grade'])  # Replace with actual clinical feature names

        return predicted_df

    except Exception as e:
        raise Exception(f"Error predicting clinical features: {e}")

def predict_survival_outcomes(clinical_features: pd.DataFrame) -> pd.DataFrame:
    """Predicts survival probabilities for treatment combinations based on clinical features."""
    global estimator_survival_clinical, scaler_survival_clinical

    try:
         # Scale the clinical features data using the loaded scaler
        scaled_data = scaler_survival_clinical.transform(clinical_features)

        # Predict survival probabilities
        survival_predictions = estimator_survival_clinical.predict(scaled_data)

        # Convert the result to a DataFrame
        survival_df = pd.DataFrame(survival_predictions, columns=['Chemo', 'Endo', 'Both', 'None'])

        return survival_df

    except Exception as e:
        raise Exception(f"Error predicting survival outcomes: {e}")

def recommend_treatment(survival_probabilities: pd.DataFrame) -> str:
    """Recommends the best treatment option based on survival probabilities."""

    try:
        # Find the treatment with the highest predicted survival probability
        best_treatment = survival_probabilities.idxmax(axis=1)[0]
        best_probability = survival_probabilities[best_treatment][0]

        # Create a justification for the recommendation
        justification = f"The model predicts the highest survival probability ({best_probability:.2f}) with {best_treatment} treatment."

        return f"Recommended treatment: {best_treatment}. {justification}"

    except Exception as e:
        raise Exception(f"Error recommending treatment: {e}")

def get_column_names(data: pd.DataFrame) -> list:
    """Returns a list of column names in the dataframe."""
    try:
        return list(data.columns)
    except Exception as e:
        raise Exception(f"Error getting column names: {e}")

def rename_columns(data:pd.DataFrame, column_mapping: list[str]) -> pd.DataFrame:
    """Renames columns in the dataframe based on the provided mapping."""
    try:
        return data.rename(columns=column_mapping)
    except Exception as e:
        raise Exception(f"Error renaming columns: {e}")
    
load_models_and_data()