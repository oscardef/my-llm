import torch
import tiktoken
from pathlib import Path

from my_llm.classification.data import download_and_prepare_data
from my_llm.classification.dataloader import get_dataloaders
from my_llm.classification.model_setup import setup_model
from my_llm.classification.training import train_classifier_simple
from my_llm.classification.plotting import plot_values
from my_llm.classification.inference import classify_review

def initialize_device():
    """
    Pick the best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main():
    # ── Settings ──────────────────────────────────────────────────────────────
    NUM_EPOCHS        = 5
    EVAL_FREQ         = 50
    EVAL_ITER         = 5
    LEARNING_RATE     = 5e-5
    WEIGHT_DECAY      = 0.1
    LOAD_EXISTING     = True   # try to load saved model if present
    MODEL_SAVE_PATH   = Path(__file__).parent / "review_classifier.pth"

    # ── Setup ─────────────────────────────────────────────────────────────────
    device    = initialize_device()
    tokenizer = tiktoken.get_encoding("gpt2")

    download_and_prepare_data()
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(tokenizer)

    model = setup_model(train_dataset, device)

    # ── Train or Load ─────────────────────────────────────────────────────────
    if LOAD_EXISTING and MODEL_SAVE_PATH.exists():
        print(f"▶ Loading existing model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("▶ Training new classifier…")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=NUM_EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER,
        )

        # save model checkpoint
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✔ Saved model to {MODEL_SAVE_PATH}")

        # plot training metrics (optional)
        plot_values(
            torch.linspace(0, NUM_EPOCHS, len(train_losses)),
            torch.linspace(0, examples_seen, len(train_losses)),
            train_losses, val_losses,
            label="loss"
        )
        plot_values(
            torch.linspace(0, NUM_EPOCHS, len(train_accs)),
            torch.linspace(0, examples_seen, len(train_accs)),
            train_accs, val_accs,
            label="accuracy"
        )

    # ── Interactive Classification Loop ──────────────────────────────────────
    print(f"\nModel ready on {device}.  Type a review to classify it, or ‘quit’ to exit.\n")
    while True:
        user_text = input("Review> ").strip()
        if not user_text or user_text.lower() in ("quit", "exit"):
            print("Exiting classifier.")
            break

        # classify_review returns e.g. "Positive" or "Negative" (or similar)
        label = classify_review(
            user_text,
            model,
            tokenizer,
            device,
            max_length=train_dataset.max_length
        )
        print(f"→ Predicted label: {label}\n")

if __name__ == "__main__":
    main()
