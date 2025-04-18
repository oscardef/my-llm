import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class GPTDatasetV1(Dataset):
    """
    A PyTorch Dataset for preparing language modeling data using a sliding window approach.

    This dataset tokenizes the provided text and creates overlapping sequences of tokens of fixed maximum length. 
    For each sequence, the input is a sequence of tokens and the target is the same sequence shifted by one token, 
    which is a common setup for training autoregressive language models.

    Attributes:
        input_ids (List[torch.Tensor]): A list of torch Tensors, each representing an input sequence of token ids.
        target_ids (List[torch.Tensor]): A list of torch Tensors, each representing the corresponding target sequence of token ids.

    Methods:
        __init__(txt: str, tokenizer, max_length: int, stride: int):
            Initializes the dataset by tokenizing the input text and constructing overlapping sequences.
            
            Args:
                txt (str): The complete text to be tokenized and split into sequences.
                tokenizer: Tokenizer object with an `encode` method that returns token ids from the text.
                max_length (int): The maximum length of each input sequence.
                stride (int): The stride or step size used for the sliding window to generate overlapping sequences.
        
        __len__():
            Returns the number of sequences in the dataset.
        
        __getitem__(idx: int):
            Retrieves the input-target pair at the given index.
            
            Args:
                idx (int): The index of the desired sequence pair.
            
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: A tuple where the first element is the input sequence tensor and the 
                second element is the corresponding target sequence tensor.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    Creates a DataLoader for training or evaluating a language model on a text dataset.

    This function initializes a GPT-2 tokenizer using tiktoken, processes the provided text into a dataset with overlapping sequences
    (using GPTDatasetV1), and then constructs a PyTorch DataLoader to iterate over the dataset in batches.

    Parameters:
        txt (str): The source text to build the dataset from.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        max_length (int, optional): The maximum sequence length for each sample. Defaults to 256.
        stride (int, optional): The step size between sequences allowing overlap. Defaults to 128.
        shuffle (bool, optional): If True, shuffles the dataset at every epoch. Defaults to True.
        drop_last (bool, optional): If True, drops the last batch if it's incomplete. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.

    Returns:
        DataLoader: A PyTorch DataLoader object providing batches of tokenized text data.
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


def assign(left, right):
    """
    Assigns the values of right to left, ensuring they have the same shape.
    Args:
        left: The tensor to assign values to.
        right: The tensor to assign values from.
    Returns:
            The left tensor with values assigned from right.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    """
    Load weights from a dictionary into the GPT model.
    Args:
        gpt: The GPT model instance.
        params: Dictionary containing the weights.
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    # Iterates over each transformer block in the model
    for b in range(len(params["blocks"])):
        # np.split function is used to divide the attention and bias weights into three equal parts for the query, key, and value components.
        q_w, k_w, v_w = np.split( 
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # The original GPT-2 model by OpenAI reused the token embedding weights in the output layer to 
    # reduce the total number of parameters, which is a concept known as weight tying.
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])