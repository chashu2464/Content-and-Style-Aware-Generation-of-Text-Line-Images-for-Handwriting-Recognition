from torch import nn

# from torch.autograd import Variable
import numpy as np
from parameters import *
import torch.functional as F
from attention import MultiHeadAttention

# from models.vgg_tro_channel1 import vgg16_bn
from models import Visual_encoder
from block import *


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        B, C, H, W = (
            batch_size,
            32,
            IMAGE_HEIGHT * scale_factor,
            IMAGE_WIDTH * scale_factor,
        )
        self.num_heads = 4
        head_size = 200
        print(f"channel:-{C=},Hight:- {H=} width:- {W=} Batch:- {B=}")
        self.resnet = Generator_Resnet(class_num=2, num_res_blocks=2)
        self.visual_encoder = Visual_encoder()  # vgg
        self.layer_norm = nn.LayerNorm([W])  #
        self.attention = MultiHeadAttention(num_heads=self.num_heads, dropout=0.4, C=C)
        self.dropout = nn.Dropout(p=0.3)
        # self.upsample=nn.Linear(head_size*self.num_heads,out_feature)
        self.linear = nn.Linear(
            in_features=W, out_features=W, bias=False
        )

    def forward(self, x):
        resent = self.resnet(x)  # resent   batch_size,outchannel,Hight , Width

        # resent=resent.view(batch_size,-1)
        visual_encder = self.visual_encoder(x)  # visual encoder for positionin
        # visual_encder=visual_encder.view(batch_size,-1)
        print(
            f"Shape of the resent output{resent.shape} and Vgg output shape{visual_encder.shape}"
        )
        combained_out = resent + visual_encder  # combained before input

        layer_norm = self.layer_norm(
            combained_out
        )  # layer_norm    batch_size features  eg 2,32,20,10 = 2,32,200
        attention = self.attention(layer_norm)  # attention layer
        dropout = self.dropout(attention)  # dropout layer
        layer_norm = layer_norm.repeat(dropout.size(0) // batch_size, 1, 1, 1)
        norm_dropout = (
            layer_norm + dropout
        )  # combained output of layer_norm and dropout
        print("Shape of the norm dropout,", norm_dropout.shape)
        layer_norm1 = self.layer_norm(norm_dropout)  # layer_norm output
        linear = self.linear(layer_norm1.view(-1,layer_norm1.size(3)))
        print("shape of the linear",linear.shape)
        linear=linear.view(layer_norm1.size(0),layer_norm1.size(1),layer_norm1.size(2),layer_norm1.size(3))
        linear_drop = self.dropout(linear)
        norm_dropout1 = linear_drop + norm_dropout
        final_norm = self.layer_norm(norm_dropout1)

        return final_norm
