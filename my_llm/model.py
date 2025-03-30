import torch
import torch.nn as nn
from helpers import GELU
from attention import MultiHeadAttention

class DummyGPTModel(nn.Module):
    def __init__(self, cfg): # cfg is a dictionary containing the model configuration
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # Token embedding, this is for the converted the input tokens into embeddings.
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # Positional embedding, this is for the position of the tokens
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # Dropout layer for the embeddings
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        """
        Describes the data flow through the model: computes token and positional embeddings for the input indicies, 
        applies dropout, processes the data through the transformer blocks, applies normalization, and finally produces logits with
        the linear output layer.
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) # Get token embeddings for the input
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # Get positional embeddings
        x = tok_embeds + pos_embeds # Add the token and positional embeddings. We do this to give the model information about the position of the tokens.
        x = self.drop_emb(x) # Apply dropout to the embeddings
        x = self.trf_blocks(x) # Apply the transformer blocks]
        x = self.final_norm(x) # Apply layer normalization
        logits = self.out_head(x) # Get logits. This is the output of the model.
        return logits # Return the logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased=False means that the variance is calculated with N instead of N-1
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

class FeedForward(nn.Module):
    """
    A simple feed-forward network with GELU activation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    

# Testing the model
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1))) # Convert the text to tokens
batch.append(torch.tensor(tokenizer.encode(txt2))) # Convert the text to tokens
batch = torch.stack(batch, dim=0) # Stack the tensors. Meaning, combine the tensors into a single tensor
print(batch)

torch.manual_seed(123)

from config import GPT_CONFIG_124M

model = DummyGPTModel(cfg=GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

             