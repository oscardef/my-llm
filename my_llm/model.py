import torch
import torch.nn as nn
from helpers import GELU
from attention import MultiHeadAttention

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # Token embedding
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # Position embedding
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # Dropout layer for embeddings
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) # Stack n transformer blocks
        
        self.final_norm = LayerNorm(cfg["emb_dim"]) # Layer normalization for the output
        self.out_head = nn.Linear( 
            cfg["emb_dim"], cfg["vocab_size"], bias=False # Linear layer for output
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape # Get the batch size and sequence length
        tok_embeds = self.tok_emb(in_idx) # Get the token embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # Arange will create a tensor with values from 0 to seq_len. Device will put the tensor on the same device as in_idx
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x) # Apply dropout to the embeddings
        x = self.trf_blocks(x) # Apply the transformer blocks
        x = self.final_norm(x) # Apply layer normalization
        logits = self.out_head(x) # Get the logits
        return logits # Shape [batch_size, num_tokens, vocab_size]

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
    

# # Testing the model
# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1))) # Convert the text to tokens
# batch.append(torch.tensor(tokenizer.encode(txt2))) # Convert the text to tokens
# batch = torch.stack(batch, dim=0) # Stack the tensors. Meaning, combine the tensors into a single tensor
# print(batch)

# torch.manual_seed(123)

# from config import GPT_CONFIG_124M

# model = GPTModel(GPT_CONFIG_124M)

# out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)