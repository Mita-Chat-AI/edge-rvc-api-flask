import os
import base64
import traceback
from io import BytesIO
from functools import lru_cache

import librosa
from loguru import logger
from pydub import AudioSegment
from flask_restx import Api, Resource
from flask import Flask, request, Response

from .utils import load_model
from .utils import hubert_model


limitation = os.getenv("SYSTEM") == "spaces"



model_root = "weights"
models = [
    d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
]
if not models:
    raise ValueError("No model found in `weights` folder")
models.sort()

app = Flask(__name__)


api = Api(app, version='1.0', title='RVC API', description='API для RVC.')
ns = api.namespace('api', description='RVC operations')


@lru_cache()
def rvc_process(
    model_name: str,
    audio_stream: BytesIO,
    f0_up_key: int,
    f0_method: str,
    index_rate: float,
    filter_radius: int,
    resample_sr: int,
    rms_mix_rate: float,
    protect: float,
):
    try:
        tgt_sr, net_g, vc, version, index_file, if_f0 = load_model(model_name)
        audio, sr = librosa.load(audio_stream, sr=16000, mono=True)

        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            sr,  # Pass the sr here
            times,
            f0_up_key,
            f0_method,
            index_file,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,  # f0_file
        )

        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr

        audio_segment = AudioSegment(
            audio_opt.tobytes(),
            frame_rate=tgt_sr,
            sample_width=2,
            channels=1,
        )

        output_stream = BytesIO()
        audio_segment.export(output_stream, format="mp3")
        output_stream.seek(0)  # Reset the buffer to the beginning

        return output_stream.read()

    except Exception as e:
        print(traceback.format_exc())
        return {'error': str(e)}, 500


from .utils import Data

@ns.route("/get_rvc")
class RVCAPI(Resource):
    @ns.expect(Data, validate=True)
    def post(self):
        """Преобразование аудио с использованием RVC (вход - bytes)."""
        json_data = request.get_json()
        data = Data(**json_data)

        try:
            audio_data_base64 = data.audio
            audio_bytes = base64.b64decode(audio_data_base64)
            audio_stream = BytesIO(audio_bytes)

            model_name = data.person
            f0_up_key = data.pith
            
            
            f0_method = data.get('f0_method', 'rmvpe')
            index_rate = float(0.75)
            filter_radius = int(data.get('filter_radius', 5))
            resample_sr = int(data.get('resample_sr', 0))
            rms_mix_rate = float(data.get('rms_mix_rate', 0.4))
            protect = float(data.get('protect', 0.7))

        except KeyError as e:
            return {"error": f"Missing parameter: {e}"}, 400
        except base64.binascii.Error as e:
            return {"error": f"Invalid base64 audio data: {e}"}, 400
        except Exception as e:
            return {"error": f"Error processing input data: {e}"}, 400


        if os.path.exists("rmvpe.pt") and os.path.exists("hubert_base.pt"):
            import time
            s = time.time()
            response = rvc_process(
                model_name,
                audio_stream,
                f0_up_key,
                f0_method,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
            )

            st = time.time()
            logger.info(f"{st - s} seconds")
            if isinstance(response, tuple):  # Error handling
                return response
            return Response(response)
        else:
            return {
                "message": "У вас не загружен rmvpe.pt или hubert_base.pt.  Запустите скрипт для их загрузки."
            }, 500
            
            

