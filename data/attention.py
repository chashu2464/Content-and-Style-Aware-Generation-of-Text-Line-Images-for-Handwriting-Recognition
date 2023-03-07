import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Q, K, V weight matrices for each head
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / (d_model ** 0.5)

        # Output projection matrix
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x shape: [batch_size, num_channels, image_height, image_width]
        batch, channel, high, width = x.shape
        # Reshape input to [batch_size, num_channels*image_height, image_width]
        x = x.view(x.size(0), -1, x.size(1))

        # Compute Q, K, V matrices for each head
        q = self.wq(x)  # q shape: [batch_size, num_channels*image_height, d_model]
        k = self.wk(x)  # k shape: [batch_size, num_channels*image_height, d_model]
        v = self.wv(x)  # v shape: [batch_size, num_channels*image_height, d_model]
        weights = torch.matmul(q, k.transpose(-2, -1))
        weights = weights * self.scale
        weights = nn.functional.softmax(weights, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(weights, v)
        return output.view(batch, channel, high, width)


class MultiHeadAttention(nn.Module):
    "Multiple heads of the self_attention in parallel"

    def __init__(self, num_heads, C, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(d_model=C, num_heads=num_heads) for _ in range(num_heads)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads])

        out = self.dropout(out)
        print(out.shape)
        return out
