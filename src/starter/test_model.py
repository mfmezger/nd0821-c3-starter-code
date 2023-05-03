import os
import sys
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)
from ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=5, n_classes=2, random_state=42
    )
    return X, y

def test_train_model(classification_data):
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    
    assert model is not None, "Model should not be None"
    assert hasattr(model, "fit"), "Model should have a 'fit' method"

def test_compute_model_metrics(classification_data):
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "F1 score should be between 0 and 1"

def test_inference(classification_data):
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray), "Predictions should be a NumPy array"
    assert len(preds) == len(y_test), "Number of predictions should match the number of test labels"