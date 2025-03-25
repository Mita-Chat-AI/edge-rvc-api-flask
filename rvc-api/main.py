import logging
import os
import traceback
from io import BytesIO
from pathlib import Path
import uuid

import librosa
import torch
from fairseq import checkpoint_utils
from flask import Flask, request, Response
from flask_cors import CORS
from loguru import logger
from pydub import AudioSegment
from flask_restx import Api, Resource, fields

from .config import Config
from .lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from .vc_infer_pipeline import VC

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global variables
hubert_model = None  # Initialize to None

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

limitation = os.getenv("SYSTEM") == "spaces"

config = Config()

model_root = "weights"
models = [
    d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
]
if not models:
    raise ValueError("No model found in `weights` folder")
models.sort()

app = Flask(__name__)
CORS(app) # Enable CORS

api = Api(app, version='1.0', title='RVC API', description='API для RVC.')
ns = api.namespace('api', description='RVC operations')


# Define input model for BYTES data
rvc_input_bytes_model = api.model('RVCInputBytes', {
    'model_name': fields.String(required=True, description='Имя модели RVC'),
    'f0_up_key': fields.Integer(default=0, description='Сдвиг высоты тона'),
    'f0_method': fields.String(default='rmvpe', description='Метод определения высоты тона'),
    'index_rate': fields.Float(default=0.75, description='Коэффициент индекса'),
    'filter_radius': fields.Integer(default=5, description='Радиус фильтра'),
    'resample_sr': fields.Integer(default=0, description='Частота передискретизации'),
    'rms_mix_rate': fields.Float(default=0.4, description='Коэффициент смешивания RMS'),
    'protect': fields.Float(default=0.7, description='Уровень защиты'),
    'audio_data': fields.String(required=True, description='Audio data as base64 encoded string')
})



def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    return hubert_model.eval()


logger.info("Loading hubert model...")
hubert_model = load_hubert()
logger.success("Hubert model loaded.")


from functools import lru_cache
import base64

@lru_cache() # use lru_cache
def load_model(model_name):
    """Loads a model and its associated VC instance.  Uses lru_cache for caching."""
    print(f"Loading model: {model_name}")

    pth_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".pth")
    ]
    if not pth_files:
        raise ValueError(f"No pth file found in {model_root}/{model_name}")

    pth_path = pth_files[0]
    print(f"Loading {pth_path} from disk")
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        net_g = (
            SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
            if if_f0 == 1
            else SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        )
    elif version == "v2":
        net_g = (
            SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
            if if_f0 == 1
            else SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
        )
    else:
        raise ValueError("Unknown version")

    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    print("Model loaded")
    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()

    vc = VC(tgt_sr, config)

    index_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".index")
    ]
    index_file = index_files[0] if index_files else ""
    print(f"Index file found: {index_file}") if index_file else print("No index file found")

    return tgt_sr, net_g, vc, version, index_file, if_f0

# Example of how to use the cached model:
# tgt_sr, net_g, vc, version, index_file, if_f0 = load_model("your_model_name")


@lru_cache() # use lru_cache
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
        audio, sr = librosa.load(audio_stream, sr=16000, mono=True)  # Загружаем аудио из BytesIO

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

        # Convert numpy array to bytes using pydub
        audio_segment = AudioSegment(
            audio_opt.tobytes(),
            frame_rate=tgt_sr,
            sample_width=2,  # 2 bytes for int16
            channels=1,  # mono
        )

        # Export to mp3 in memory
        output_stream = BytesIO()
        audio_segment.export(output_stream, format="mp3")
        output_stream.seek(0)  # Reset the buffer to the beginning

        return output_stream.read()

    except Exception as e:
        print(traceback.format_exc())
        return {'error': str(e)}, 500


@ns.route("/get_rvc")
class RVCResourceBytes(Resource):
    @api.expect(rvc_input_bytes_model)
    def post(self):
        """Преобразование аудио с использованием RVC (вход - bytes)."""
        data = request.get_json()
        if not data:
            return {"error": "No data provided"}, 400

        try:
            audio_data_base64 = data['audio_data']
            audio_bytes = base64.b64decode(audio_data_base64)
            audio_stream = BytesIO(audio_bytes)

            model_name = data['model_name']
            f0_up_key = int(data.get('f0_up_key', 0))
            f0_method = data.get('f0_method', 'rmvpe')
            index_rate = float(data.get('index_rate', 0.75))
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
            
            

