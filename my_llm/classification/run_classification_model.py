import torch
import tiktoken
from pathlib import Path

from my_llm.classification.data import download_and_prepare_data
from my_llm.classification.dataloader import get_dataloaders
from my_llm.classification.model_setup import setup_model
from my_llm.classification.training import train_classifier_simple
from my_llm.classification.plotting import plot_values
from my_llm.classification.inference import classify_review

if __name__ == "__main__":
    # Settings
    NUM_EPOCHS = 5
    EVAL_FREQ = 50
    EVAL_ITER = 5
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.1
    LOAD_EXISTING_MODEL = True  # Set True to load saved model
    MODEL_SAVE_PATH = Path(__file__).parent / "review_classifier.pth"

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    download_and_prepare_data()
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(tokenizer)

    model = setup_model(train_dataset, device)

    if LOAD_EXISTING_MODEL and MODEL_SAVE_PATH.exists():
        print(f"Loading existing model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("Training new model...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=NUM_EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER,
        )

        # Save the newly trained model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved model to {MODEL_SAVE_PATH}")

        # Save plots
        plot_values(
            torch.linspace(0, NUM_EPOCHS, len(train_losses)),
            torch.linspace(0, examples_seen, len(train_losses)),
            train_losses,
            val_losses,
            label="loss"
        )

        plot_values(
            torch.linspace(0, NUM_EPOCHS, len(train_accs)),
            torch.linspace(0, examples_seen, len(train_accs)),
            train_accs,
            val_accs,
            label="accuracy"
        )

    # Run a few examples
    print(classify_review("Hey, just wanted to check if we're still on for dinner tonight? Let me know!", model, tokenizer, device, max_length=train_dataset.max_length))
    print(classify_review("You are a winner you have been specially selected to receive $1000 cash or a $2000 award.", model, tokenizer, device, max_length=train_dataset.max_length))
