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
    global multi_output_model_selected, scaler_selected, clinical_features_predicted

    try:
        # Get genes in correct order
        gene_expression_data = gene_expression_data[selected_genes]
        # Scale the gene expression data using the loaded scaler
        scaled_data = scaler_selected.transform(gene_expression_data)
        scaled_data = pd.DataFrame(scaled_data, columns=selected_genes)
        # Predict clinical features
        predicted_clinical_features = multi_output_model_selected.predict(scaled_data)

        # Convert the result to a DataFrame (assuming your model returns a numpy array)
        predicted_df = pd.DataFrame(predicted_clinical_features, columns=clinical_features_predicted)  

        return predicted_df

    except Exception as e:
        raise Exception(f"Error predicting clinical features: {e}")

def predict_survival_outcome(clinical_features: pd.DataFrame, chemo=False, endo=False, days=365) -> list:
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
         # Scale the clinical features data using the loaded scaler
        #scaled_data = scaler_survival_clinical.transform(clinical_features)
        clinical_features = clinical_features[clinial_features_survival]
        clinical_features['endocrine treated'] = endo
        clinical_features['chemo treated'] = chemo
        # Predict survival probabilities
        survival_predictions = estimator_survival_clinical.predict_cumulative_hazard_function(clinical_features)
        survival_probabilities = [np.exp(-chf(days)) for chf in survival_predictions]
        return survival_probabilities

    except Exception as e:
        raise Exception(f"Error predicting survival outcomes: {e}")

def predict_survival_outcomes(data: pd.DataFrame, days=365) -> pd.DataFrame:
    """Predicts survival probabilities for various treatment combinations based on clinical features."""
    global estimator_survival_clinical, scaler_survival_clinical, clinial_features_survival

    try:

        # Predict survival probabilities for all treatment combinations
        survival_probabilities = pd.DataFrame({
            'no treatment': predict_survival_outcome(data, chemo=False, endo=False, days=days),
            'endocrine treatment': predict_survival_outcome(data, chemo=False, endo=True, days=days),
            'chemotherapy': predict_survival_outcome(data, chemo=True, endo=False, days=days),
            'chemotherapy + endocrine treatment': predict_survival_outcome(data, chemo=True, endo=True, days=days)
        })

        return survival_probabilities

    except Exception as e:
        raise Exception(f"Error predicting survival outcomes: {e}")

def recommend_treatment(clinical_features: pd.DataFrame) -> str:
    """Recommends the best treatment option based on survival probabilities."""

    try:
        survival_probabilities = predict_survival_outcomes(clinical_features)
        # Find the treatment with the highest predicted survival probability
        best_treatment = survival_probabilities.idxmax(axis=1)
        best_probability = survival_probabilities

        # Create a justification for the recommendation

        return  (best_treatment, best_probability)

    except Exception as e:
        raise Exception(f"Error recommending treatment: {e}")

def get_column_names(data: pd.DataFrame) -> list:
    """Returns a list of first 10 column names in the dataframe."""
    try:
        return list(data.columns)[:10]
    except Exception as e:
        raise Exception(f"Error getting column names: {e}")

def rename_columns(data:pd.DataFrame, column_mapping: list[str]) -> pd.DataFrame:
    """Renames columns in the dataframe based on the provided mapping."""
    try:
        return data.rename(columns=column_mapping)
    except Exception as e:
        raise Exception(f"Error renaming columns: {e}")
    
load_models_and_data()