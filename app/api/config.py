from pydantic import BaseSettings
from typing import List
import os

class Settings (BaseSettings):
    API_NAME = "hdb_prediction"
    API_STR: str = os.environ.get("API_STR", "/api/v1")
    MODEL_PATH: str = os.environ.get("MODEL_PATH", "app/api/router/v1/tuned_lgbm_19k.pkl")
    MODEL_NAME: str = os.environ.get("MODEL_NAME", "lgbm")
    MODEL_TAG: str = os.environ.get("MODEL_TAG", "Tuned LGBM 19K RMSE")
    DATA_PATH: str = os.environ.get("MODEL_PATH", "data/hdb_latest.csv")

    # DIMENSION_REDUCTION: str = os.environ.get("DIMENSION_REDUCTION")
    # CLUSTERING: str = os.environ.get("CLUSTERING")
    # ADDITIONAL_FEATURES_OUT: list[str] = os.environ.get("ADDITIONAL_FEATURES_OUT")
    # DIMENSION_REDUCTION: str = os.environ.get("DIMENSION_REDUCTION")

    MODEL_CONFIG : dict = os.environ.get("MODEL_CONFIG", 
            {
                'training_size':0.5,
                'split': False,
                'dimension_reduction': None,
                'clustering': None,
                'additional_features_out': None,
                'random_state': 42
            }
        )

    # LIMIT: int = os.environ.get("LIMIT", 500000)
    # DATA_GOV_API: str = os.environ.get("DATA_GOV_API", "https://data.gov.sg/api/action/datastore_search") 
    # WRITE_FILE_NAME:str = os.environ.get("WRITE_FILE_NAME", "latest_data.csv")
    ONEMAP_API:str = os.environ.get("ONEMAP_API", "https://developers.onemap.sg/commonapi/search")
    MRT_API:str = os.environ.get("MRT_API", "https://datamall.lta.gov.sg/content/dam/datamall/datasets/PublicTransportRelated/Train%20Station%20Codes%20and%20Chinese%20Names.zip")
    MRT_EXCEL_FILE:str = os.environ.get("MRT_EXCEL_FILE", "Train Station Codes and Chinese Names.xls")

SETTINGS = Settings()