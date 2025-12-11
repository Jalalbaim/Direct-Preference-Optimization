#IMPORTS -----------------
import os
import sys

import torch
from torch.utils.data import DataLoader

# pour que "src" soit dans le path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dpo.models import load_models
from src.dpo.data import PreferenceDataset, preference_collate_fn
from src.dpo.trainer import DPOTrainer
from src.dpo.utils import load_yaml_config


#MAIN FUNCTIONS -----------------
def main():
    config_path = "configs/sentiment.yaml"
    config = load_yaml_config(config_path)

    model_name = config["model"]["name"]
    dtype = config["model"]["dtype"]

    # modèles + tokenizer
    mb = load_models(model_name, dtype=dtype)
    tokenizer = mb.tokenizer