from pydantic import BaseModel, Field


class Data(BaseModel):
    text: str = Field(required=True, description='Текст для генерации', min_length=10, max_length=800)
    person: str = Field(required=True, min_length=5, max_length=50, description="Персонаж")
    rate: str = Field(required=True, min_length=3 , max_length=4, description="скорость речи")
    pith: int = Field(required=True, ge=-10, le=+15, description="Высота тона")