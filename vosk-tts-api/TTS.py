
import base64
import requests
from io import BytesIO

from loguru import logger
from .config import config


from vosk_tts import Model, Synth

model = Model(model_name="vosk-model-tts-ru-0.7-multi")
synth = Synth(model)

class TTS:
    def __init__(self):
        ()
        
    @staticmethod
    def vosk(
        text: str,
        speaker_id: int,
        speech_rate: str,
        duration_noise_level: str,
        scale: str,
    ) -> BytesIO:
        try:
            # communicate = edge_tts.Communicate(text=text, rate=rate)
            # audio_data = BytesIO()

            # for chunk in communicate.stream_sync():
            #     if chunk["type"] == "audio":
            #         audio_data.write(chunk["data"])
            # audio_data.seek(0)



            wav_bytes = synth.synth_bytes(
                text=text,
                speaker_id=speaker_id,
                speech_rate=speech_rate,
                duration_noise_level=duration_noise_level,
                scale=scale
            )

            audio_data = BytesIO(wav_bytes)
            audio_data.seek(0)


            return audio_data
        except Exception as e:
            logger.error(f"TTS-ERROR {e}")
            return None

    @staticmethod
    def rvc(
        person: str,
        text: str,
        speaker_id: int,
        speech_rate: str,
        duration_noise_level: str,
        scale: str,
        pith: int
    ) -> BytesIO | None:
        try:
            wav_buffer = TTS.vosk(
                text,
                speaker_id,
                speech_rate,
                duration_noise_level,
                scale
            )
            if not wav_buffer:
                return None

            # wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            payload = {
                "person": person,
                "pith": pith,
                "audio": audio_base64
            }

            headers = {'Content-type': 'application/json'}
            response = requests.post(
                config.rvc_api,
                json=payload,
                headers=headers,


            )
            response.raise_for_status()

            return response.content

        except requests.exceptions.RequestException as e:
            logger.error(f"RVC-API ERROR: {e}")
            raise
        except Exception as e:
            logger.error(f"Audio-conversion ERROR: {e}")
            return None

