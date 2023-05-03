"""Script to train machine learning model."""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


def main():
    # Add code to load in the data.
    data = pd.read_csv("starter/data/census.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)
    FILE_PATH = "starter/model/"

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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train and save a model.
    ada_boost = train_model(X_train, y_train)

    model_path = os.path.join(FILE_PATH, "ada_boost.pkl")
    encoder_path = os.path.join(FILE_PATH, "encoder.pkl")
    lb_path = os.path.join(FILE_PATH, "lb.pkl")

    # Save AdaBoost model
    with open(model_path, "wb") as model_file:
        pickle.dump(ada_boost, model_file)

    # Save OneHotEncoder
    with open(encoder_path, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)

    # Save LabelBinarizer
    with open(lb_path, "wb") as lb_file:
        pickle.dump(lb, lb_file)

    # Predict
    preds = inference(ada_boost, X_test)

    print(
        "precision: {}, recall: {}, fbeta: {}".format(*compute_model_metrics(y_test, preds))
    )


if __name__ == "__main__":
    main()