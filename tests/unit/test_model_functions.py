# test_model_functions.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from agent import model_functions as mf

# Mocking the model loading to avoid actual file loading during tests
@pytest.fixture(autouse=True)
def mock_load_models():
    with patch('agent.model_functions.load_models_and_data') as mock:
        yield mock

# Sample data for testing
@pytest.fixture
def sample_gene_expression_data():
    return pd.DataFrame(np.random.rand(5, len(mf.selected_genes)), columns=mf.selected_genes)

@pytest.fixture
def sample_clinical_features_data():
    return pd.DataFrame(np.random.rand(5, len(mf.clinincal_variables_needed)), columns=mf.clinincal_variables_needed)

# Test cases for validate_data
def test_validate_data_gene_expression_valid(mock_load_models, sample_gene_expression_data):
    assert mf.validate_data(sample_gene_expression_data, 'gene_expression') == True

def test_validate_data_gene_expression_invalid(mock_load_models):
    invalid_data = pd.DataFrame({'wrong_gene': [1, 2, 3]})
    with pytest.raises(ValueError):
        mf.validate_data(invalid_data, 'gene_expression')

def test_validate_data_clinical_features_valid(mock_load_models, sample_clinical_features_data):
    assert mf.validate_data(sample_clinical_features_data, 'clinical_features') == True

def test_validate_data_clinical_features_invalid(mock_load_models):
    invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
    with pytest.raises(ValueError):
        mf.validate_data(invalid_data, 'clinical_features')

def test_validate_data_invalid_data_type(mock_load_models):
    data = pd.DataFrame({'gene1': [1, 2, 3]})
    with pytest.raises(ValueError):
        mf.validate_data(data, 'invalid_type')

# Test cases for predict_clinical_features
@patch('agent.model_functions.multi_output_model_selected')
@patch('agent.model_functions.scaler_selected')
def test_predict_clinical_features(mock_scaler, mock_model, sample_gene_expression_data):
    mock_scaler.transform.return_value = sample_gene_expression_data.values  # Mock scaling
    mock_model.predict.return_value = np.random.rand(5, len(mf.clinical_features_predicted))  # Mock prediction
    result = mf.predict_clinical_features(sample_gene_expression_data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, len(mf.clinical_features_predicted))

@patch('agent.model_functions.multi_output_model_selected')
@patch('agent.model_functions.scaler_selected')
def test_predict_clinical_features_error(mock_scaler, mock_model, sample_gene_expression_data):
    mock_scaler.transform.side_effect = Exception("Scaling error")
    with pytest.raises(Exception, match="Error predicting clinical features"):
        mf.predict_clinical_features(sample_gene_expression_data)

# Test cases for predict_survival_outcome
@patch('agent.model_functions.estimator_survival_clinical')
def test_predict_survival_outcome(mock_estimator, sample_clinical_features_data):
    # Mock the predict_cumulative_hazard_function method
    mock_estimator.predict_cumulative_hazard_function.return_value = [lambda t: 0.1 * t] * len(sample_clinical_features_data)  # Mock CHF

    result = mf.predict_survival_outcome(sample_clinical_features_data)
    assert isinstance(result, list)
    assert len(result) == len(sample_clinical_features_data)
    assert all(0 <= p <= 1 for p in result)  # Probabilities should be between 0 and 1

@patch('agent.model_functions.estimator_survival_clinical')
def test_predict_survival_outcome_error(mock_estimator, sample_clinical_features_data):
    mock_estimator.predict_cumulative_hazard_function.side_effect = Exception("Prediction error")
    with pytest.raises(Exception, match="Error predicting survival outcomes"):
        mf.predict_survival_outcome(sample_clinical_features_data)

# Test cases for predict_survival_outcomes
@patch('agent.model_functions.predict_survival_outcome')
def test_predict_survival_outcomes(mock_predict_outcome, sample_clinical_features_data):
    # Mock predict_survival_outcome to return a fixed probability for each treatment
    mock_predict_outcome.return_value = [0.8] * len(sample_clinical_features_data)

    result = mf.predict_survival_outcomes(sample_clinical_features_data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(sample_clinical_features_data), 4)  # 4 treatment options

@patch('agent.model_functions.predict_survival_outcome')
def test_predict_survival_outcomes_error(mock_predict_outcome, sample_clinical_features_data):
    mock_predict_outcome.side_effect = Exception("Prediction error")
    with pytest.raises(Exception, match="Error predicting survival outcomes"):
        mf.predict_survival_outcomes(sample_clinical_features_data)

# Test cases for recommend_treatment
@patch('agent.model_functions.predict_survival_outcomes')
def test_recommend_treatment(mock_predict_outcome, sample_clinical_features_data):
    # Mock predict_survival_outcomes to return a DataFrame with fixed probabilities
    mock_data = {'no treatment': [0.1] * len(sample_clinical_features_data),
                 'endocrine treatment': [0.2] * len(sample_clinical_features_data),
                 'chemotherapy': [0.3] * len(sample_clinical_features_data),
                 'chemotherapy + endocrine treatment': [0.4] * len(sample_clinical_features_data)}
    mock_predict_outcome.return_value = pd.DataFrame(mock_data)

    best_treatment, best_proba = mf.recommend_treatment(sample_clinical_features_data)
    assert isinstance(best_treatment, pd.Series)
    assert isinstance(best_proba, pd.DataFrame)
    assert all(treatment == "chemotherapy + endocrine treatment" for treatment in best_treatment)

@patch('agent.model_functions.predict_survival_outcomes')
def test_recommend_treatment_error(mock_predict_outcome, sample_clinical_features_data):
    mock_predict_outcome.side_effect = Exception("Prediction error")
    with pytest.raises(Exception, match="Error recommending treatment"):
        mf.recommend_treatment(sample_clinical_features_data)

def test_get_column_names():
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]})
    result = mf.get_column_names(data)
    assert result == ['col1', 'col2', 'col3']

def test_get_column_names_error():
    with pytest.raises(Exception, match="Error getting column names"):
        mf.get_column_names(None)

def test_rename_columns():
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    column_mapping = {'col1': 'new_col1', 'col2': 'new_col2'}
    result = mf.rename_columns(data, column_mapping)
    assert list(result.columns) == ['new_col1', 'new_col2']

def test_rename_columns_error():
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    column_mapping = {'col1': 'new_col1', 'col3': 'new_col3'}  # 'col3' does not exist
    with pytest.raises(Exception, match="Error renaming columns"):
        mf.rename_columns(data, column_mapping)