import torch
import torch.nn as nn
from my_llm.base_model.helpers import calc_loss_batch, calc_loss_loader
from my_llm.base_model.helpers import generate_and_print_sample

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    Trains a model using a simple training loop with periodic evaluation and text generation.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print sample after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate the performance of the provided model using training and validation data loaders.

    This function sets the model to evaluation mode, computes the loss over a specified number 
    of batches from both the training and validation datasets using the calc_loss_loader function, 
    and then resets the model to training mode. This ensures that layers such as dropout or batch 
    normalization behave appropriately during evaluation.

    Parameters:
        model (torch.nn.Module): The neural network model to evaluate.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader providing the validation dataset.
        device (torch.device): The computation device (e.g., CPU or GPU).
        eval_iter (int): The number of batches to process from each DataLoader for evaluation.

    Returns:
        tuple: A tuple containing:
            - train_loss (float): The computed loss over the training data.
            - val_loss (float): The computed loss over the validation data.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss