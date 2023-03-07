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
        B,C,H,W=batch_size,32,IMAGE_HEIGHT*scale_factor,IMAGE_WIDTH*scale_factor
        self.num_heads=2
        head_size=200
        out_feature=C*H*W
        print(f"channel:-{C=},Hight:- {H=} width:- {W=} Batch:- {B=}")
        self.resnet=Generator_Resnet(class_num=2,num_res_blocks=2)
        self.visual_encoder=Visual_encoder() #vgg 
        self.layer_norm=nn.LayerNorm([H*W*C]) #, 10*20*32
        self.attention=MultiHeadAttention(num_heads=self.num_heads,H=H,W=W,B=B,C=C,scale_factor=scale_factor,head_size=head_size,dropout=0.4)
        self.dropout=nn.Dropout(p=0.3)
        self.upsample=nn.Linear(head_size*self.num_heads,out_feature)
        self.linear=nn.Linear(in_features=out_feature,out_features=out_feature,bias=False)

    def forward(self,x):
        resent=self.resnet(x)   #resent   batch_size,outchannel,Hight , Width    

        resent=resent.view(batch_size,-1)
        visual_encder=self.visual_encoder(x)  #visual encoder for positionin
        visual_encder=visual_encder.view(batch_size,-1)
        print(f"Shape of the resent output{resent.shape} and Vgg output shape{visual_encder.shape}")
        combained_out=resent+visual_encder #combained before input
        
        layer_norm=self.layer_norm(combained_out) #layer_norm    batch_size features  eg 2,32,20,10 = 2,32,200
        attention=self.attention(layer_norm)  #attention layer
        dropout=self.dropout(attention)   #dropout layer
        upsample_attention=self.upsample(dropout.view(2,-1))
        norm_dropout=layer_norm+upsample_attention   #combained output of layer_norm and dropout
        print("Shape of the norm dropout,",norm_dropout[0].shape)
        layer_norm1=self.layer_norm(norm_dropout)  #layer_norm output
        linear=self.linear(layer_norm1)     #linear layer
        linear_drop=self.dropout(linear)  
        norm_dropout1=linear_drop+norm_dropout
        final_norm=self.layer_norm(norm_dropout1)

        return final_norm




        
    


