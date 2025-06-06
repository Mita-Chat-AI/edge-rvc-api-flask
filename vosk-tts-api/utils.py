from pydantic import BaseModel, Field


class Data(BaseModel):
    text: str = Field(required=True, description='Текст для генерации', min_length=10, max_length=800)
    person: str = Field(required=True, min_length=5, max_length=50, description="Персонаж")
    rate: str = Field(required=True, min_length=3, max_length=4, description="Скорость речи")
    pith: int = Field(required=True, ge=-10, le=15, description="Высота тона")
    speaker_id: int = Field(default=2, ge=0, le=10, description="ID спикера")
    speech_rate: float = Field(default=1.0, ge=0.5, le=2.0, description="Темп речи")
    duration_noise_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Уровень шума длительности")
    scale: float = Field(default=1.0, ge=0.5, le=2.0, description="Масштаб модели")
