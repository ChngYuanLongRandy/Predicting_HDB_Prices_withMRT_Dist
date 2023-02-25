"""
Instantiate models and other requisites here
"""
import pandas as pd
from app.api.config import SETTINGS
from src.datapipeline import Modelpipeline

def load_dataset(file_path:str):
    df = pd.read_csv(file_path)
    return df

def prep_pipeline(model_path:str, params:dict):
    pipe = Modelpipeline(DATASET)
    pipe.preprocess(training_size=params['training_size'], split=params['split'])
    pipe.load_model(model_path)
    return pipe

DATASET = load_dataset(SETTINGS.DATA_PATH)
PIPELINE = prep_pipeline(SETTINGS.MODEL_PATH, SETTINGS.MODEL_CONFIG)
