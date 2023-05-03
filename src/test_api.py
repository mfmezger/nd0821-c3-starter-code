import os
import sys
import json
from fastapi.testclient import TestClient

file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the API!"}


def test_predict_sub_50():
    # create a sample input for the model
    sample_input = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert response.text.strip() in ["<=50K", ">50K"]

def test_predict_over_50():
    # create a sample input for the model
    sample_input = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 217400000,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert response.text == ">50K"

def test_predict_failure():
    # create a sample input for the model
    sample_input = {
        "age": 39,
        "workclass": "State-gov",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 500