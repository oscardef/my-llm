import torch
import re
from my_llm.base_model.model import GPTModel
from my_llm.base_model.dataloader import load_weights_into_gpt
from my_llm.base_model.gpt_download import download_and_load_gpt2

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def setup_model(device, choose_model="gpt2-medium (355M)"):
    """
    Load GPT-2 model with pretrained weights.
    """
    model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")

    model_config = BASE_CONFIG.copy()
    model_config.update(model_configs[choose_model])

    model = GPTModel(model_config)
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    load_weights_into_gpt(model, params)

    model.eval()
    model.to(device)

    return model
