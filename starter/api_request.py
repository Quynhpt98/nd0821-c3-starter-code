import json
import requests
data = {
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States",
    }
response = requests.post("http://127.0.0.1:8000/predict/", json=data)

print("status_code response", response.status_code)
print("data response: salary", response.json()["salary"] )