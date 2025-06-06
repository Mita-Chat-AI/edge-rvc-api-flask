import os
from functools import lru_cache

import torch
from  loguru import logger
from fairseq import checkpoint_utils
from pydantic import BaseModel, Field

from .rmvpe import RMVPE
from  .config import rvconfig as config
from .lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)


model_root = "weights"
models = [
    d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
]
if not models:
    raise ValueError("No model found in `weights` folder")
models.sort()

class Data(BaseModel):
    person: str = Field(required=True, min_length=1, max_length=50, description="Персонаж")
    pith: int = Field(required=True, ge=-10, le=+15, description="Высота тона")
    audio: bytes = Field(required=True, description="поток байтов")


@lru_cache()
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
    
    from .vc_infer_pipeline import VC

    vc = VC(tgt_sr, config)

    index_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".index")
    ]
    index_file = index_files[0] if index_files else ""
    print(f"Index file found: {index_file}") if index_file else print("No index file found")

    return tgt_sr, net_g, vc, version, index_file, if_f0


import os
import requests

HUBERT_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

def download_model_if_missing(filename: str, url: str):
    if not os.path.exists(filename):
        print(f"Файл '{filename}' не найден. Скачиваем с {url} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Скачивание '{filename}' завершено.")
    else:
        print(f"Файл '{filename}' уже существует.")

def download_hubert_and_rmvpe():
    download_model_if_missing("hubert_base.pt", HUBERT_URL)
    download_model_if_missing("rmvpe.pt", RMVPE_URL)



def load_hubert():
    global hubert_model

    if not os.path.exists("hubert_base.pt"):
        download_hubert_and_rmvpe()

    print("Загружаем модель Hubert...")

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ['hubert_base.pt'],
        suffix="",
    )

    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    hubert_model.eval()

    print("Hubert модель успешно загружена.")
    return hubert_model

def load_rvmpe():
    global model_rmvpe

    if not os.path.exists("rmvpe.pt"):
        download_hubert_and_rmvpe()

    model_rmvpe = None
    logger.info("Loading rmvpe model...")
    try:
        
        model_rmvpe = RMVPE("rmvpe.pt", is_half=config.is_half, device=config.device)
        logger.success("rmvpe model loaded.")
        return model_rmvpe
    except Exception as e:
        logger.error(f"Failed to load rmvpe model: {e}")
        exit()

model_rmvpe = load_rvmpe()
hubert_model = load_hubert()



