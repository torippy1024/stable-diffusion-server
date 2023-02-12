from pydantic import BaseModel, Field


class ApiDiffusionGetResponse(BaseModel):
    url: str = Field("", description="generated image's data (base64)")
