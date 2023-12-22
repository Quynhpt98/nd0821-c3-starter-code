import pandas as pd
from starter.ml.model import train_model, compute_model_metrics
from starter.ml.data import process_data
import logging
file_path = './data/census_clean.csv'
# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)
data = data[:20]
label = 'salary'
# Setup logging
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",

]
X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
)
logging.basicConfig(filename='tests/logging.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s'
                    )


def test_train_model():
    try:
        _ = train_model(X, y)
        logging.info("Test train model succeed")
    except Exception:
        logging.error("Test train_model fail!")
        raise AssertionError("Can not train model")


def test_inference():
    try:
        model = train_model(X, y)
        _ = model.predict(X)
        logging.info("Test inference succeed")
    except Exception:
        logging.error("Test inference fail!")
        raise AssertionError("Can not inference")


def test_compute_model_metrics():
    try:
        model = train_model(X, y)
        predictions = model.predict(X)
        compute_model_metrics(y, predictions)
        logging.info("Test compute_model_metrics succeed")
    except Exception:
        logging.error("Test compute_model_metrics fail!")
        raise AssertionError("Can not compute_model_metrics")
