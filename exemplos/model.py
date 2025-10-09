from einops import rearrange
import torch
import torch.nn as nn

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, head_dim, dropout):
        super().__init__()
        self.head_dim = head_dim
        self.fc_q = nn.Linear(d_model, head_dim)
        self.fc_k = nn.Linear(d_model, head_dim)
        self.fc_v = nn.Linear(d_model, head_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([head_dim])))

    def forward(self, x, mask=None):
        Q, K, V = self.fc_q(x), self.fc_k(x), self.fc_v(x)
        qV = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            qV = qV.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(qV, dim=-1)
        return torch.matmul(self.dropout(attention), V) 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0

        self.heads = nn.ModuleList([
            SingleHeadAttention(d_model, d_model // n_heads, dropout)
            for _ in range(n_heads)
        ])

        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        head_outputs = [head(x, mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.fc_out(concatenated)

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        return self.norm2(x + self.dropout(ff_output))
    
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        patch_vector_dim = patch_size * patch_size * in_channels
        
        self.projection = nn.Linear(patch_vector_dim, d_model)

    def forward(self, x):
        # x shape: [batch_size, in_channels, height, width]
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=self.patch_size, p2=self.patch_size)

        return self.projection(x)
