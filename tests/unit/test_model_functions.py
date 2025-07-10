# test_model_functions.py
import pytest
from agent import model_functions as mf
import pytest
import pandas as pd
import numpy as np
from io import StringIO
import sys, os
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path.endswith("tests/unit"):
    # change path to tests directory
    sys.path.append("../../app")
    sys.path.append("../../tests")
if current_path.endswith("app"):
    sys.path.append("../tests")


# Mock gene list and clinical columns
MOCK_GENES = ['gene1', 'gene2', 'gene3']
MOCK_CLINICAL = ['er status', 'pgr status', 'her2 status', 'ki67 status', 'nhg']

@pytest.fixture(autouse=True)
def load_models(monkeypatch):
    # Patch models and scalers
    monkeypatch.setattr(mf, 'selected_genes', MOCK_GENES)
    monkeypatch.setattr(mf, 'clinical_features_survival', MOCK_CLINICAL)

    class DummyScaler:
        def transform(self, X): return np.array(X)

    class DummyModel:
        def predict(self, X): return np.ones((len(X), 8))  # for 8 clinical predictions
        def predict_cumulative_hazard_function(self, X):
            return [lambda t: 0.01 * t for _ in range(len(X))]

    monkeypatch.setattr(mf, 'scaler_selected', DummyScaler())
    monkeypatch.setattr(mf, 'scaler_survival_clinical', DummyScaler())
    monkeypatch.setattr(mf, 'multi_output_model_selected', DummyModel())
    monkeypatch.setattr(mf, 'estimator_survival_clinical', DummyModel())

def test_validate_data_gene_expression(monkeypatch):
    csv = StringIO("gene1,gene2,gene3\n1,2,3\n4,5,6")
    monkeypatch.setattr(pd, 'read_csv', lambda _: pd.read_csv(csv))
    assert mf.validate_data("dummy.csv", "gene_expression") is True

def test_validate_data_clinical(monkeypatch):
    csv = StringIO("er status,pgr status,her2 status,ki67 status,nhg\n+, +, +, +, +")
    monkeypatch.setattr(pd, 'read_csv', lambda _: pd.read_csv(csv))
    assert mf.validate_data("dummy.csv", "clinical_features") is True

def test_validate_data_invalid_type():
    with pytest.raises(ValueError):
        mf.validate_data("dummy.csv", "invalid_type")

def test_predict_clinical_features(monkeypatch, tmp_path):
    csv = StringIO("gene1,gene2,gene3\n1,2,3\n4,5,6")
    monkeypatch.setattr(pd, 'read_csv', lambda _: pd.read_csv(csv))
    out_path = mf.predict_clinical_features("dummy.csv")
    assert out_path == "user/predicted_features.csv"

def test_predict_survival_outcome(monkeypatch):
    csv = StringIO("er status,pgr status,her2 status,ki67 status,nhg\n1,1,1,1,1\n1,1,1,1,1")
    monkeypatch.setattr(pd, 'read_csv', lambda _: pd.read_csv(csv))
    result = mf.predict_survival_outcome("dummy.csv", chemo=True, endo=True, days=365)
    assert all(0 < x <= 1 for x in result)

def test_predict_survival_outcomes(monkeypatch):
    csv = StringIO("er status,pgr status,her2 status,ki67 status,nhg\n1,1,1,1,1\n1,1,1,1,1")
    monkeypatch.setattr(pd, 'read_csv', lambda _: pd.read_csv(csv))
    result = mf.predict_survival_outcomes("dummy.csv", days=365)
    assert set(result.keys()) == {
        'no treatment', 'endocrine treatment', 'chemotherapy', 'chemotherapy + endocrine'
    }

def test_get_column_names(monkeypatch):
    df = pd.DataFrame(columns=[f"col{i}" for i in range(20)])
    monkeypatch.setattr(pd, 'read_csv', lambda _: df)
    cols = mf.get_column_names("dummy.csv")
    assert len(cols) == 10

def test_rename_columns(monkeypatch, tmp_path):
    df = pd.DataFrame({"a": [1], "b": [2]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(pd, 'read_csv', lambda _: pd.read_csv(path))
    result_path = mf.rename_columns(str(path), {"a": "x", "b": "y"})
    df_renamed = pd.read_csv(result_path)
    assert list(df_renamed.columns) == ["x", "y"]

def test_read_data_from_csv(monkeypatch):
    df = pd.DataFrame({"x": [1, 2, 3]})
    monkeypatch.setattr(pd, 'read_csv', lambda _: df)
    result = mf.read_data_from_csv("dummy.csv")
    assert result.equals(df)
