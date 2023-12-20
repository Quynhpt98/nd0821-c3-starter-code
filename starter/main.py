# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from starter.ml.model import inference
from starter.ml.data import process_data
import uvicorn
import joblib


# Load your CSV file
# Define Pydantic model for POST request body
class InputData(BaseModel):
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    # salary: str
# Initialize FastAPI


app = FastAPI(swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"})


# Root endpoint - GET
@app.get("/")
def read_root():
    return {"message": "Welcome to the model inference API!"}


# Model inference endpoint - POST
@app.post("/predict/")
def predict(data: InputData):
    # Access data sent in the request using the InputData model
    # Example usage:
    input_dict = {
        "workclass": [data.workclass],
        "education": [data.education],
        "marital_status": [data.marital_status],
        "occupation": [data.occupation],
        "relationship": [data.relationship],
        "race": [data.race],
        "sex": [data.sex],
        "native_country": [data.native_country],
        "age": [12],
        "fnlgt": [123],
        "education_num": [123],
        "capital_gain": [123],
        "capital_loss": [123],
        "hours_per_week": [2],
    }
    cats = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    input_data = pd.DataFrame(input_dict)
    # print(input_data)
    loaded_model = joblib.load('starter/model/trained_model.pkl')
    encoder = joblib.load('starter/model/encoder.pkl')
    lb = joblib.load('starter/model/lb.pkl')
    X, _, _, _ = process_data(
            X=input_data,
            categorical_features=cats,
            training=False,
            encoder=encoder,
            lb=lb)
    predict = inference(loaded_model, X)
    status_code = 200
    result = {"salary": str(predict[0]), "status_code": status_code}
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
