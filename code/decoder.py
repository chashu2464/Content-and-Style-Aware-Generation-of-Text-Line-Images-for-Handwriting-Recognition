import torch
from torch import nn
from attention import MultiHeadAttention, MultiHead_CrossAttention
from models import TextEncoder_FC
from parameters import (
    embedding_size,
    text_max_len,
    batch_size,
    device,
    number_feature,
    Max_str
    

)
from load_data import NUM_CHANNEL,IMG_HEIGHT,IMG_WIDTH

input_feature=NUM_CHANNEL*IMG_WIDTH*IMG_HEIGHT

class Decorder(torch.nn.Module):
    def __init__(self, in_feature=IMG_HEIGHT*IMG_WIDTH, out_feature=text_max_len, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.TextStyle = TextEncoder_FC(Max_str).to(device)
        self.in_feature = embedding_size *IMG_HEIGHT*IMG_WIDTH

        self.linear_upsampling = nn.Linear(
            number_feature, input_feature
        )
        self.linear_downsampling = nn.Linear(
            in_features=input_feature, out_features=self.out_feature
        )
        ##self.upsample_norm=nn.Conv2d(batch_size*num_heads,NUM_CHANNEL*num_heads,1,1,)

        self.block_with_attention = LayerNormLinearDropoutBlock(
            in_features=input_feature,
            out_features=self.out_feature,
            num_heads=2,
            dropout_prob=0.2,
            attention=True,
        )
        self.block_without_attention = LayerNormLinearDropoutBlock(
            in_features=input_feature,
            out_features=self.out_feature,
            num_heads=2,
            dropout_prob=0.2,
            attention=False,
        )
        self.norm = nn.LayerNorm(self.out_feature)
        self.cross_attention = MultiHead_CrossAttention(
            infeature=self.out_feature,
            out_feature=self.out_feature,
            num_heads=1,
            dropout=0.2,
        )
        self.drop = nn.Dropout(self.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out,text_style_content=None,img_shape=None,):

        # char_embedding= 2,5440 global_net=[2,64,10,100]  batch,output,H,W
        char_embedding, global_net = self.TextStyle(text_style_content,img_shape)

        char_upsampling = self.linear_upsampling(char_embedding)
        txt_style = global_net + char_upsampling.view(
            global_net.size(0),
            global_net.size(1),
            global_net.size(2),
            global_net.size(3),
        )
        ## layer norm has the same shape are hte txt_style and global_net [2,64000]
        # attention_block=2,85
        print("in deco")

        attetion_block, layer_norm = self.block_with_attention(
            txt_style.reshape(txt_style.size(0), -1)
        )
        norm_down_sample = self.linear_downsampling(layer_norm)
        norm_down_sample = norm_down_sample.repeat(
            attetion_block.size(0) // batch_size, 1
        )# mask for  text 

        attention_norm = attetion_block + norm_down_sample
        block_without_attention, _ = self.block_without_attention(attention_norm)

        combained_without_attention = block_without_attention + attention_norm

        norm = self.norm(combained_without_attention)

        #upsample_norm=self.upsample_norm(norm.unsqueeze(1)).squeeze(1)
        norm=norm.repeat(encoder_out.size(0)//norm.size(0),1)

        cross_attention = self.cross_attention(norm, encoder_out)
        drop_out = self.drop(cross_attention)
        # norm = norm.repeat(drop_out.size(0) // (batch_size + batch_size), 1)
        combained_without_attention = drop_out + norm

        block_without_attention2, _ = self.block_without_attention(
            combained_without_attention
        )
        final_combained = block_without_attention2 + combained_without_attention

        soft_max = self.softmax(final_combained)
        return final_combained


class LayerNormLinearDropoutBlock(nn.Module):
    def __init__(
        self, in_features, out_features, num_heads, dropout_prob=0.1, attention=False
    ):
        super(LayerNormLinearDropoutBlock, self).__init__()
        self.attention = attention
        # Define the layer norm, linear layer, and dropout modules
        if self.attention:
            self.layer_norm = nn.LayerNorm(in_features)
            self.atten = MultiHeadAttention(
                in_features, out_features, num_heads, dropout_prob
            )
        else:
            self.layer_norm = nn.LayerNorm(out_features)
            self.linear = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Apply layer norm to the input tensor
        layer_norm = self.layer_norm(x)
        if self.attention:
            x = self.atten(layer_norm)
        else:
            # Apply linear transformation to the input tensor


            x = self.linear(layer_norm)

        # Apply dropout to the output of the linear layer
        x = self.dropout(x)

        return x, layer_norm
