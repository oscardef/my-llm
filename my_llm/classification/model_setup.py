import torch
from my_llm.base_model.model import GPTModel
from my_llm.base_model.dataloader import load_weights_into_gpt
from my_llm.base_model.gpt_download import download_and_load_gpt2

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
}

def setup_model(train_dataset, device, model_name="gpt2-small (124M)"):
    """
    Load a pretrained GPT model and modify it for classification.
    """
    BASE_CONFIG.update(MODEL_CONFIGS[model_name])

    model_size = model_name.split(" ")[-1].strip("()")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], 2)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    model.to(device)
    return model