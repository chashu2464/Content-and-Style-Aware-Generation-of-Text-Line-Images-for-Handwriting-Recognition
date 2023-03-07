import torch 
from torch import nn
from attention import MultiHeadAttention
from models import TextEncoder_FC
from parameters import embedding_size,text_max_len,IMAGE_HEIGHT,IMAGE_WIDTH,batch_size
class Decorder(torch.nn.Module):
    def __init__(self,in_feature=32,out_feature=32,dropout=0.3):
        super().__init__()
        self.dropout=dropout
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.TextStyle=TextEncoder_FC()
        # self.linear_upsampling=nn.Linear(embedding_size*text_max_len,embedding_size*IMAGE_WIDTH*IMAGE_HEIGHT)
        # self.layer_norm=nn.LayerNorm(IMAGE_WIDTH)  # batch_size,sequence_len and embedding_dim , in_feature shape should be embedding_dim
        # self.mutli_attention = MultiHeadAttention(num_heads=4,dropout=0.4,C=embedding_size)
        # self.dropout=nn.Dropout(self.dropout)
        # self.linear=nn.Linear(in_features=embedding_size*IMAGE_HEIGHT*IMAGE_WIDTH,out_features=100)
        self.block_with_attention=LayerNormLinearDropoutBlock(in_features=embedding_size*IMAGE_HEIGHT*IMAGE_WIDTH,out_features=128,num_heads=2,dropout_prob=0.2,attention=True)
    def forward(self,x):

        char_embedding,global_net=self.TextStyle(x)
        char_upsampling=self.linear_upsampling(char_embedding)
        txt_style=global_net+char_upsampling.view(global_net.size(0),global_net.size(1),global_net.size(2),global_net.size(3))
        print(f"{txt_style.shape=}")
        attetion_block=self.block_with_attention(txt_style)
        print(f"{attetion_block.shape=}")
        # layer_norm=self.layer_norm(txt_style)
        # attention=self.mutli_attention(layer_norm)  #[2, 64, 15, 300] batch,embedding_size,
        # dropout=self.dropout(attention)
        # layer_norm = layer_norm.repeat(dropout.size(0) // batch_size, 1, 1, 1)

        # drop_norm=layer_norm+attention
        # layer_norm1=self.layer_norm(drop_norm)    #[8, 64, 15, 300] [batch, embedding_size,width,hight]
        # print(f"{attention.shape=} {layer_norm1.shape=}") #[8, 64, 15, 300]

        # linear=self.linear(layer_norm1.view(layer_norm1.size(0),-1))
        # linear_drop=self.dropout(linear)
        # print("linear",linear_drop.shape)

class LayerNormLinearDropoutBlock(nn.Module):
    def __init__(self, in_features, out_features,num_heads, dropout_prob=0.1,attention=False):
        super(LayerNormLinearDropoutBlock, self).__init__()
        self.attention=attention
        # Define the layer norm, linear layer, and dropout modules
        self.layer_norm = nn.LayerNorm(in_features)
        if self.attention:
            self.atten=MultiHeadAttention(in_features,out_features,num_heads,dropout_prob)
        else:
            print("attention is not applied")
            self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Apply layer norm to the input tensor
        layer_norm = self.layer_norm(x)
        if self.attention:
             x=self.atten(layer_norm)
        else:        
        # Apply linear transformation to the input tensor
            
            print("attention is not applied")

            x = self.linear(layer_norm)

        # Apply dropout to the output of the linear layer
        x = self.dropout(x)

        return x,layer_norm