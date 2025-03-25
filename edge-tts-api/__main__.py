from loguru import logger
from flask_restx import Api, Resource
from flask import Flask, request, Response

from .config import config
from .TTS import TTS
from .utils import Data


app = Flask(__name__)

api = Api(app, version='1.0', title='TTS API', description='TTS API')
ns = api.namespace('api/v1/edge', description='TTS-edge operations')


@ns.route("/get_edge")
class GetEdge(Resource):
    @ns.expect(Data, validate=True)
    def post(self):
        try:
            json_data = request.get_json()
            data = Data(**json_data)

            text = data.text
            person = data.person
            rate = data.rate
            pith = data.pith

            audio_data = TTS.rvc(
                text,
                person,
                rate,
                pith
            )

            if audio_data:
                return Response(audio_data, mimetype="audio/mpeg")
            else:
                return {"error": "Ошибка при преобразовании аудио"}, 500

        except Exception as e:
            logger.error(f"GET-EDGE ERROR: {e}")
            return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=False)