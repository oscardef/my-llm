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


def initialize_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

def initialize_model(device):
    model_name = "gpt2-small (124M)"
    model_config = GPT_CONFIG_124M.copy()
    model_config.update(model_configs[model_name])
    model_config.update({"context_length": 1024, "qkv_bias": True})
    
    gpt = model.GPTModel(model_config)
    gpt.eval()

    print("Loading weights...")
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    print("Weights loaded.")
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    return gpt, model_config

def interactive_prompt_loop(gpt, tokenizer, device, model_config):
    while True:
        user_input = input("Enter a prompt (type 'quit' to exit): ")
        if user_input.lower() == "quit":
            print("Exiting...")
            break

        token_ids = generate(
            model=gpt,
            idx=text_to_token_ids(user_input, tokenizer).to(device),
            max_new_tokens=25,
            context_size=model_config["context_length"],
            top_k=50,
            temperature=1.5
        )
        print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    # Main execution
    tokenizer = tiktoken.get_encoding("gpt2")
    device = initialize_device()
    print(f"Using {device} device.")
    gpt, model_config = initialize_model(device)
    torch.manual_seed(123)
    interactive_prompt_loop(gpt, tokenizer, device, model_config)


    