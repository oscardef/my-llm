import torch
import torch.nn as nn    

def text_to_token_ids(text, tokenizer):
    """
    Convert a text string into a tensor of token IDs using the provided tokenizer.

    Parameters:
        text (str): The input string to be tokenized.
        tokenizer: An instance of a tokenizer that implements an 'encode' method. The encode method should support an
            'allowed_special' parameter to specify which special tokens are allowed (in this case, '<|endoftext|>').

    Returns:
        torch.Tensor: A tensor containing the token IDs of the encoded text with an added batch dimension.
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs to text.
    Args:
        token_ids: Tensor of token IDs.
        tokenizer: Tokenizer to decode the token IDs.
    Returns:
        decoded_text: Decoded text.
    """
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate text using the model.
    Args:
        model: The model to be used for generation.
        idx: Input token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        context_size: Size of the context window.
        temperature: Temperature for sampling (default is 0.0).
        top_k: Number of top logits to consider for sampling (default is None).
        eos_id: End-of-sequence token ID (default is None).
    Returns:
        idx: Generated token IDs.
    """
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1] # Get the minimum value of the top_k logits
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using the model with a simple greedy approach.
    Args:
        model: The model to be used for generation.
        idx: Input token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        context_size: Size of the context window.
    Returns:
        idx: Generated token IDs.
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

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