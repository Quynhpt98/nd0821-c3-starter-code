from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Test GET endpoint


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the model inference API!"}

# Test POST endpoint for model inference - Scenario 1


def test_predict_scenario_one():
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
    response = client.post("/predict/", json=data)

    assert response.status_code == 200
    assert response.json()["salary"] == '0'


# Test POST endpoint for model inference - Scenario 2
def test_predict_scenario_two():
    data = {
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married"
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == 422
    assert response.json() is not None
