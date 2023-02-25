from datetime import datetime
from pydantic import BaseModel


class FlatType(BaseModel):
    text: str = "4 ROOM"


class PostalCode(BaseModel):
    text: int = 560325


class StoreyRange(BaseModel):
    text: str = "01 TO 03"

class Input(BaseModel):
    flat_type : FlatType
    postal_code: PostalCode
    storey_range: StoreyRange