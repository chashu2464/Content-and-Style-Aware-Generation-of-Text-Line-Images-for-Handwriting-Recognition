from torch import nn

# from torch.autograd import Variable
import numpy as np
from parameters import *
import torch.nn.functional as F

# from models.vgg_tro_channel1 import vgg16_bn
from models import ImageEncoder
from simple_resnet import ResidualBlock
from decoder import LayerNormLinearDropoutBlock
from load_data import IMG_HEIGHT,IMG_WIDTH,NUM_CHANNEL
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # B, C, H, W = (
        #     batch_size,
        #     32,
        #     IMAGE_HEIGHT * scale_factor,
        #     IMAGE_WIDTH * scale_factor,
        # )
        self.num_layer=8
        self.num_heads = 2
        self.in_feature = IMG_HEIGHT*IMG_WIDTH*batch_size
        self.out_feature = text_max_len
        #self.resnet = Generator_Resnet(num_res_blocks=2).to(device)
        self.resnet=self.res_blocks = nn.Sequential(
            *[ResidualBlock(2 * 2**i, 2 * 2**(i+1)) for i in range(self.num_layer)]
        )
        #self.visual_encoder = Visual_encoder().to(device)  # vgg
        self.visual_encoder=ImageEncoder().to(device=device)
        self.norm_down = nn.Linear(
            in_features=IMG_HEIGHT*IMG_WIDTH*512, out_features=self.out_feature
        )
        self.upsample_norm=nn.Conv2d(batch_size,NUM_CHANNEL*self.num_heads,1,1,)

        self.block_with_attention = LayerNormLinearDropoutBlock(
            in_features=IMG_HEIGHT*IMG_WIDTH*512,
            out_features=self.out_feature,
            num_heads=self.num_heads,
            dropout_prob=0.2,
            attention=True,
        )
        self.block_without_attention = LayerNormLinearDropoutBlock(
            in_features=IMG_HEIGHT*IMG_WIDTH*batch_size,
            out_features=self.out_feature,
            num_heads=self.num_heads,
            dropout_prob=0.2,
            attention=False,
        )
        self.norm = nn.LayerNorm(self.out_feature)

    def forward(self, x):
        resnet = self.resnet( #shape matches
            x.permute(1,0,2,3)
        )  # resent   batch_size,outchannel,Hight , Width

        # resent=resent.view(batch_size,-1)
        visual_encode = self.visual_encoder(x)  # visual encoder for positionin
        # visual_encder=visual_encder.view(batch_size,-1)
        resnet_shape=resnet.shape
        resized_vis = F.interpolate(visual_encode, size=resnet_shape[2:], mode='bilinear', align_corners=False)
        repeat=resnet.size(0)//visual_encode.size(0)
        
        resized_vis=resized_vis.repeat(*(repeat,1,1,1))

        combained_out = resnet + resized_vis  # combained before input
        combained_out = F.interpolate(combained_out, size=[32,80], mode='bilinear', align_corners=False)

        attention_block, norm_layer = self.block_with_attention(
            combained_out.view(combained_out.size(0), -1)
        )

        down_sampled_norm = self.norm_down(norm_layer)
        #up_sample_norm=self.upsample_norm(down_sampled_norm.unsqueeze(1))

        down_sampled_norm = down_sampled_norm.repeat(
            attention_block.size(0) // down_sampled_norm.size(0), 1
        )

        combained_attention = down_sampled_norm + attention_block

        without_attention, _ = self.block_without_attention(combained_attention)
        combained_with_attention = combained_attention + without_attention
        final_norm = self.norm(combained_with_attention) #4,32
        return final_norm
