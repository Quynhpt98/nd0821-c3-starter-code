import pandas as pd
from ml.model import silce_performance
from ml.data import process_data

from sklearn.model_selection import train_test_split
import joblib
import json
# Replace 'file_path.csv' with the path to your CSV file
file_path = 'data/census_clean.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

train, test = train_test_split(data, test_size=0.20)

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
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=True
)
loaded_model = joblib.load('model/trained_model.pkl')
encoder = joblib.load('model/encoder.pkl')
lb = joblib.load('model/lb.pkl')
label = "salary"
res = silce_performance(test, cat_features, label, encoder, lb, loaded_model)
file = open("slice_output.txt", "w")
json.dump(res, file, indent=4)
file.close()
