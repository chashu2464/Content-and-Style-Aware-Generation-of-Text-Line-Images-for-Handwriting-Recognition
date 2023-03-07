import torch
from torch import nn
import torch.functional as F
from parameters import *
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, H,W,C,B,head_size,dropout):
        super().__init__()
        final_out=32
        print(f"In features in linear keys in Attention",H*W*C)
        # print(f"Scale factor values {scale_factor=}")
        # Width,Height=W*scale_factor,H*scale_factor
        # print(f"{Height=} and {Width=}")
        infeature=H*W*C
        outfeature=200
        self.key = nn.Linear(infeature, outfeature, bias=False)
        self.query = nn.Linear(infeature,outfeature, bias=False)
        self.value = nn.Linear(infeature, outfeature, bias=False)
        self.scale = 1.0 / (head_size ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print(f" Shape of the input in the attention layer:-{x.shape}")
          
        query = self.query(x)
        key = self.key(x.view(batch_size,-1))

        # Compute attention weights using dot product
        weights = torch.matmul(query, key.transpose(-2, -1))
        weights = weights * self.scale
        weights = nn.functional.softmax(weights, dim=-1)

        # Apply attention weights to values
        value=self.value(x.view(batch_size,-1))
        output = torch.matmul(weights,value )
        print("shape of the head ", output.shape)

        return output

class MultiHeadAttention(nn.Module):
  "Multiple heads of the self_attention in parallel"
  def __init__(self,num_heads,H,W,B,C,scale_factor,head_size,dropout):
    super().__init__()
    self.heads=nn.ModuleList([Head(H=H,W=W,B=B,C=C,head_size=head_size,dropout=dropout) for _ in range(num_heads)])
    self.dropout=nn.Dropout(dropout)

  def forward(self,x):
    out= torch.cat([h(x) for h in self.heads])

    out=self.dropout (out)
    return out
  

  # 2,256,10,20
  # 2,256,80,160