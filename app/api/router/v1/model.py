from fastapi import status
from fastapi import APIRouter
import pandas as pd
from typing import Dict, List
from app.api.config import SETTINGS
from app.api.schema import Input
from app.api.resources import DATASET, PIPELINE
from src.fetch_resources import fetch_resources
import logging

ROUTER = APIRouter()

Prediction = float
Predictions = List[float]
APIVersion = Dict[str, str]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ROUTER.get("/version", status_code=status.HTTP_200_OK)
def get_api_version() -> APIVersion:
    """
    GET method for API version
    """
    return {"API_Version": SETTINGS.API_STR}

@ROUTER.post('/predict', status_code=status.HTTP_200_OK)
def predict(input: Input) -> Prediction:
    """Predict the resale price of a HDB flat

    Returns:
        Prediction: float
    """
    input_dict = input.dict()
    input_dict = {
        'postal_code': input_dict['postal_code']['text'],
        'storey_range':input_dict['storey_range']['text'],
        'flat_type':input_dict['flat_type']['text']
        }

    complete_input = fetch_resources(input_dict,DATASET)
    if complete_input is not None:
        return PIPELINE.predict(complete_input)
    else:
        return None