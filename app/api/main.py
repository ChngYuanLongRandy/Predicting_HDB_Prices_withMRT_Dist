from fastapi import FastAPI, APIRouter
from app.api.config import SETTINGS
from app.api.router.v1.model import ROUTER

APP = FastAPI(title=SETTINGS.API_NAME)
API_ROUTER = APIRouter()
API_ROUTER.include_router(ROUTER, prefix="/"+SETTINGS.MODEL_NAME)#, tags=SETTINGS.MODEL_TAG)
APP.include_router(API_ROUTER, prefix=SETTINGS.API_STR)