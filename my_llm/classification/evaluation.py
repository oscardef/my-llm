import torch

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate cross-entropy loss for a single batch.

    Args:
        input_batch (Tensor): Input batch tensor.
        target_batch (Tensor): Target labels tensor.
        model (nn.Module): Model to evaluate.
        device (torch.device): Device to use.

    Returns:
        loss (Tensor): Loss value.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate average loss over a data loader.

    Args:
        data_loader (DataLoader): Data loader.
        model (nn.Module): Model to evaluate.
        device (torch.device): Device to use.
        num_batches (int, optional): Max number of batches to use.

    Returns:
        float: Average loss.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    Calculate accuracy over a data loader.

    Args:
        data_loader (DataLoader): Data loader.
        model (nn.Module): Model to evaluate.
        device (torch.device): Device to use.
        num_batches (int, optional): Max number of batches to use.

    Returns:
        float: Accuracy (between 0 and 1).
    """
    model.eval()
    correct, total = 0, 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches and i >= num_batches:
            break
        preds = torch.argmax(model(input_batch.to(device))[:, -1, :], dim=-1)
        correct += (preds == target_batch.to(device)).sum().item()
        total += target_batch.size(0)
    return correct / total


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate model loss on training and validation datasets.

    Args:
        model (nn.Module): Model to evaluate.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to use.
        eval_iter (int): Number of batches to evaluate.

    Returns:
        tuple: (train_loss, val_loss)
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
