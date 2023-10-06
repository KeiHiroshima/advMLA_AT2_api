from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

from catboost import CatBoostRegressor, Pool

"""
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

from models import LSTMModel
"""


app = FastAPI()
model_cat = load("../models/catboost_100iterations_womovingaverage.joblib")
# model_cat = load("../models/catboost_100iterations_womovingaverageid.joblib")
# model_lstm = load("../models/lstm_win365_lr0001.joblib")


@app.get("/")
def read_root():
    return {
        "description": "This is the machine learning model.",
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "Displayproject information, list of endpoints, expected parameters, and output format.",
                "parameters": None,
                "output_format": "JSON",
            },
            "/health": {
                "method": "GET",
                "description": "Welcome message",
                "parameters": None,
                "output_format": "JSON",
            },
            "/sales/national": {
                "method": "GET",
                "description": "Returning next 7 days sales volume forecast for an input date.",
                "parameters": "date, event_name, event_type, moving_average_for_90_days, moving_average_for_365_days",
                "output_format": "JSON",
            },
            "sales/stores/items": {
                "method": "GET",
                "description": "Returning predicted sales volume for an input item, store and date.",
                "parameters": "item_id, dept_id, cat_id, store_id, state_id, event_name, event_type, movave_7, movave_90, movave_365",
                "output_format": "JSON",
            },
        },
        "GitHub_repo": {
            "modeling": "https://github.com/KeiHiroshima/advMLA_AT2",
            "api": "https://github.com/KeiHiroshima/advMLA_AT2",
        },
    }


@app.get("/health", status_code=200)
def healthcheck():
    return "predictive/forecasting models are all ready to go."


def format_features_national(
    date: str,
    event_name: str,
    event_type: str,
):
    return {
        "date": [date],
        "event_name": [event_name],
        "event_type": [event_type],
    }


def format_features_items(
    date: str,
    item_id: str,
    dept_id: str,
    cat_id: str,
    store_id: str,
    state_id: str,
    event_name: str,
    event_type: str,
):
    return {
        # "date": [date],
        "item_id": [item_id],
        "dept_id": [dept_id],
        "cat_id": [cat_id],
        "store_id": [store_id],
        "state_id": [state_id],
        "event_name": [event_name],
        "event_type": [event_type],
    }


@app.get("/sales/national")
def predict(
    date: str,
    event_name: str,
    event_type: str,
):
    features = format_features_national(
        date,
        event_name,
        event_type,
    )

    obs = pd.DataFrame(features)
    # pred = model_lstm.predict(obs)
    return {"Hello": "World"}
    # return JSONResponse(pred.tolist())


@app.get("/sales/stores/items")
def predict(
    date: str,
    item_id: str,
    dept_id: str,
    cat_id: str,
    store_id: str,
    state_id: str,
    event_name: str,
    event_type: str,
):
    features = format_features_items(
        date,
        item_id,
        dept_id,
        cat_id,
        store_id,
        state_id,
        event_name,
        event_type,
    )

    obs = pd.DataFrame(features)
    pred = model_cat.predict(obs)
    return JSONResponse(pred.tolist())
