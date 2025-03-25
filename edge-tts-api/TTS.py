
import base64
import requests
from io import BytesIO

from loguru import logger
from pydub import AudioSegment
import edge_tts

from .config import config

class TTS:
    def __init__(self):
        ()
        
    @staticmethod
    def edge(
        text: str,
        rate: str
    ) -> BytesIO:
        mp3_buffer = BytesIO()
        try:
            communicate = edge_tts.Communicate(text=text, rate=rate)
            audio_data = BytesIO()

            for chunk in communicate.stream_sync():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])
            audio_data.seek(0)

            # audio = AudioSegment.from_file(audio_data, format="mp3")
            # audio.export(mp3_buffer, format="mp3")
            # mp3_buffer.seek(0)

            return audio_data
        except Exception as e:
            logger.error(f"TTS-ERROR {e}")
            return None

    @staticmethod
    def rvc(
        text: str,
        person: str,
        rate: str,
        pith: int
    ) -> BytesIO | None:
        try:
            wav_buffer = TTS.edge(text, rate)
            if not wav_buffer:
                return None
            
            #return wav_buffer

            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')


            payload = {
                "model_name": person,
                "f0_up_key": pith,
                "audio_data": audio_base64
            }

            headers = {'Content-type': 'application/json'}
            response = requests.post(
                config.rvc_api,
                json=payload,
                headers=headers,


            )
            response.raise_for_status() # Важно!  Поднимает исключение для не-200 кодов

            return response.content # Возвращаем байты

        except requests.exceptions.RequestException as e:
            logger.error(f"RVC-API ERROR: {e}")
            raise
        except Exception as e:
            logger.error(f"Audio-conversion ERROR: {e}")
            return None

