from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
import joblib
from .data import process_data
# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()

    # Fit the model to the training data
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    fbeta = fbeta_score(y_test, predictions, beta=0.5)
    print(f"fbeta: {fbeta}")
    joblib.dump(model, 'starter/model/trained_model.pkl')
    print("saved model: starter/model/trained_model.pkl")
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def silce_performance(data, categorical_features, label, encoder, lb, model):
    results = {}
    for categorical_feature in categorical_features:
        for val in data[categorical_feature].unique():
            X = data[data[categorical_feature] == val]
            print(X)
            X_test, y_test, encoder, lb = process_data(
                X=X,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb
            )

            # print(X_test.shape)
            preds = model.predict(X_test)
            prec, recall, fbeta = compute_model_metrics(y_test, preds)
            results[f"{categorical_feature}_{val}"] = f"Precision: {prec:.1f},\
                        Recall: {recall:.1f}, fbeta: {fbeta:.1f}"
    return results
