import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention implements a multi-head self-attention mechanism commonly used in transformer architectures.
    Attributes:
        d_out (int): Output feature dimension, which must be divisible by `num_heads`.
        num_heads (int): Number of parallel attention heads.
        head_dim (int): Dimensionality for each attention head (computed as d_out // num_heads).
        W_query (torch.nn.Linear): Linear layer to project input tokens to query vectors.
        W_key (torch.nn.Linear): Linear layer to project input tokens to key vectors.
        W_value (torch.nn.Linear): Linear layer to project input tokens to value vectors.
        out_proj (torch.nn.Linear): Linear layer to combine outputs from all heads.
        dropout (torch.nn.Dropout): Dropout layer applied to the attention weights.
        mask (torch.Tensor): Triangular causal mask to prevent attention to future tokens.
    Parameters:
        d_in (int): Dimensionality of the input features.
        d_out (int): Dimensionality of the output features; must be divisible by num_heads.
        context_length (int): Maximum allowable sequence length used for creating the causal mask.
        dropout (float): Dropout probability applied to the attention weights.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, include bias terms in the linear projections for Q, K, and V.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the multi-head self-attention for the input tensor.
            The method projects the input to queries, keys, and values, reshapes them to split into
            multiple heads, computes the scaled dot-product attention with a causal mask, and finally
            projects the concatenated attention outputs back to the output dimension.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, num_tokens, d_out), representing the
                              aggregated context from all attention heads.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, 
        # this will result in errors in the mask creation further below. 
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forwar

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec