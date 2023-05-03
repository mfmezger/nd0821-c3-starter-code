import os

import logging
import sys
from pathlib import Path
import pickle
import pandas as pd
from data import process_data
from model import compute_model_metrics, inference


def slice_metrics(model, encoder, lb, data: pd.DataFrame, slice_feature: str, categorical_features: list = []) -> None:
    """Compute model metrics on slices of data

    :param model: _description_
    :type model: _type_
    :param encoder: _description_
    :type encoder: _type_
    :param lb: _description_
    :type lb: _type_
    :param data: _description_
    :type data: pd.DataFrame
    :param slice_feature: _description_
    :type slice_feature: str
    :param categorical_features: _description_, defaults to []
    :type categorical_features: list, optional
    """
    logger = logging.getLogger(__name__)
    logger.info("Performance on slices of data based on %s", slice_feature)
    X, y, _, _ = process_data(
        data, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    print("X shape: ", X.shape)
    preds = inference(model, X)

    for slice_value in data[slice_feature].unique():
        slice_index = data.index[data[slice_feature] == slice_value]
        logger.info("%s = %s", slice_feature, slice_value)
        logger.info("data size: %d", len(slice_index))
        metrics = compute_model_metrics(y[slice_index], preds[slice_index])
        logger.info("precision: %f, recall: %f, fbeta: %f", *metrics)
        logger.info("-------------------------------------------------")


if __name__ == "__main__":
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("slice_output.log"),
        ],
    )
    logger = logging.getLogger(__name__)


    data_path =  os.path.join("../..","data", "census.csv")
    data = pd.read_csv(data_path)

    model_path = os.path.join("../..","model", "ada_boost.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    encoder_path = os.path.join("../..","model", "encoder.pkl")
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    lb_path = os.path.join("../..","model", "lb.pkl")
    with open(lb_path, "rb") as f:
        lb = pickle.load(f)

    slice_metrics(model, encoder, lb, data, "education", categorical_features=cat_features)