from loguru import logger
from flask_restx import Api, Resource
from flask import Flask, request, Response

from .config import config
from .TTS import TTS
from .utils import Data


app = Flask(__name__)

api = Api(app, version='1.0', title='TTS API', description='TTS API')
ns = api.namespace('api/v1/vosk', description='TTS-vosk operations')


@ns.route("/get_vosk")
class GetEdge(Resource):
    @ns.expect(Data, validate=True)
    def post(self):
        try:
            json_data = request.get_json()
            data = Data(**json_data)
            person = data.person
            text = data.text
            speaker_id = data.speaker_id
            speech_rate = data.speech_rate
            duration_noise_level = data.duration_noise_level
            scale = data.scale
            pith = data.pith





            audio_data = TTS.rvc(
                person,
                text,
                speaker_id,
                speech_rate,
                duration_noise_level,
                scale,
                pith
            )

            if audio_data:
                return Response(audio_data, mimetype="audio/mpeg")
            else:
                return {"error": "Ошибка при преобразовании аудио"}, 500

        except Exception as e:
            logger.error(f"GET-VOSK ERROR: {e}")
            return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=False)