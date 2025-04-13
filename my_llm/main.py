import torch
import model
from config import GPT_CONFIG_124M, model_configs
import helpers
import train
import tiktoken
import numpy as np
from dataloader import load_weights_into_gpt
from gpt_download import download_and_load_gpt2
from helpers import text_to_token_ids, token_ids_to_text, generate



if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.")

    

    model_name = "gpt2-small (124M)"
    # Load the model configuration
    model_config = GPT_CONFIG_124M.copy()
    model_config.update(model_configs[model_name])
    model_config.update({"context_length": 1024, "qkv_bias": True})
    
    gpt = model.GPTModel(model_config)

    gpt.eval()

    # Load the weights
    print("Loading weights...")
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    print("Weights loaded.")
    load_weights_into_gpt(gpt, params)
    # Load the weights into the model
    gpt.to(device)
    torch.manual_seed(123)

    token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=model_config["context_length"],
    top_k=50,
    temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


    