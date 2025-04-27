import torch
from my_llm.base_model.train import train_model_simple
from my_llm.classification.plotting import plot_losses

def train_instruction_model(model, train_loader, val_loader, optimizer, device,
                             num_epochs, eval_freq, eval_iter,
                             start_context, tokenizer):
    """
    Wrapper for training an instruction fine-tuning model.
    Plots the loss after training.

    Returns:
        train_losses, val_losses, tokens_seen
    """
    train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter,
    start_context, tokenizer
    )

    # Plot losses
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    return train_losses, val_losses, tokens_seen