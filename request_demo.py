import requests

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

# make a POST request to the endpoint
response = requests.post("https://demo-qqxr.onrender.com:10000/predict", json=sample_input)
print(response.text)