import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a single batch of data.
    Args:
        input_batch: Input data batch.
        target_batch: Target data batch.
        model: The model to be evaluated.
        device: The device to perform the computation on.
    Returns:
        loss: The calculated loss.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss over a data loader.
    Args:
        data_loader: Data loader containing the dataset.
        model: The model to be evaluated.
        device: The device to perform the computation on.
        num_batches: Number of batches to evaluate. If None, evaluates all batches.
    Returns:
        average_loss: The average loss over the specified number of batches.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches # Average loss over the specified number of batches