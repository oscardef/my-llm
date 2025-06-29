import torch
import tiktoken
import time
import json
import re
from pathlib import Path
from torch import tensor

from my_llm.instruction.data import download_and_load_file, split_dataset, format_input
from my_llm.instruction.dataloader import create_dataloaders
from my_llm.instruction.model_setup import setup_model
from my_llm.instruction.training import train_instruction_model
from my_llm.instruction.ollama_evaluation import check_if_running, generate_model_scores
from my_llm.base_model.helpers import generate, text_to_token_ids, token_ids_to_text
from my_llm.classification.plotting import plot_losses

MODEL_SAVE_PATH = Path("gpt2-medium355M-sft.pth")
LOAD_EXISTING_MODEL = True  # <-- Set to True to load model without retraining


if __name__ == "__main__":
    # Settings
    FILE_PATH = "instruction-data.json"
    URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    BATCH_SIZE = 8
    NUM_EPOCHS = 2
    EVAL_FREQ = 5
    EVAL_ITER = 5

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available(): # Was not working due to memory issues
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # Download and prepare data
    data = download_and_load_file(FILE_PATH, URL)
    train_data, val_data, test_data = split_dataset(data)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, tokenizer, BATCH_SIZE, device
    )

    # Model setup
    model = setup_model(device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    if LOAD_EXISTING_MODEL and MODEL_SAVE_PATH.exists():
        print(f"Loading existing model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("Model loaded. Skipping training.")
    else:
        print("Training new model...")
        torch.manual_seed(123)
        start_context = format_input(val_data[0])

        start_time = time.time()

        train_losses, val_losses, tokens_seen = train_instruction_model(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=NUM_EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER,
            start_context=start_context, tokenizer=tokenizer
        )

        end_time = time.time()
        print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")

        # Save model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved as {MODEL_SAVE_PATH}")

        # Plot losses
        epochs_seen = tensor(range(1, len(train_losses) + 1))
        plot_losses(
            epochs_seen=epochs_seen,
            tokens_seen=tensor(tokens_seen),
            train_losses=train_losses,
            val_losses=val_losses,
            save_path=Path(__file__).parent / "plots" / "loss_plot.png"
        )



    # Test inference
    for entry in test_data[:3]:
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=1024,
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text}")
        print("-"*50)

    # Generate responses for full test set
    for i, entry in enumerate(test_data):
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=1024,
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    # Save test outputs
    with open("instruction-data-with-response.json", "w") as f:
        json.dump(test_data, f, indent=4)
    print("Saved responses to instruction-data-with-response.json")

    # Evaluate using Ollama
    if not check_if_running("ollama"):
        raise RuntimeError("Ollama not running. Please start Ollama server!")

    print("Evaluating model responses with external LLM...")
    scores = generate_model_scores(test_data, "model_response")
    print(f"Scored {len(scores)} entries.")
    print(f"Average score: {sum(scores)/len(scores):.2f}")