from datetime import datetime
from pydantic import BaseModel

class Town(BaseModel):
    text: str = "ANG MO KIO"

class FlatType(BaseModel):
    text: str = "3 ROOM"

class FlatModel(BaseModel):
    text: str = "Improved"

class FloorAreaSqm(BaseModel):
    text: float = 69.0

class StreetName(BaseModel):
    text: str = "ANG MO KIO AVE 4"


class Month(BaseModel):
    # I am not sure how to validate a date,
    # I'll put it as a string and see what happens
    text: str = f"{datetime.now().year}-{datetime.now().month}"

class LeaseCommenceDate(BaseModel):
    text: int = 1987


class StoreyRange(BaseModel):
    text: str = "07 TO 09"

class 