from typing import Optional, Tuple
from pip import main

import torch
from torch import nn
from modules import MultiHeadedAttention,PositionwiseFeedForward,ConvolutionModule,Conv2dSubsampling4,GlobalCMVN
from Processor import make_pad_mask


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        out_size: int,
        attentionheads: int,
        cnn_kernel: int,
        dropout=.1, 
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = MultiHeadedAttention(attentionheads,out_size,dropout)
        self.feed_forward = PositionwiseFeedForward(out_size,out_size,dropout,)
        self.conv_module = ConvolutionModule(out_size,cnn_kernel)
        self.norm_ff = nn.LayerNorm(out_size, eps=1e-5)  # for the FNN module
        self.norm_mha = nn.LayerNorm(out_size, eps=1e-5)  # for the MHA module
        self.ff_scale = 1.0
        self.norm_conv = nn.LayerNorm(out_size,
                                          eps=1e-5)  # for the CNN module
        self.norm_final = nn.LayerNorm(
                out_size, eps=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(dropout)
        self.size = out_size
        self.normalize_before = False
        self.concat_linear = nn.Identity()


    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x_att = self.self_attn(
            x, x, x,mask)
        x = residual + self.dropout(x_att)
        x = self.norm_mha(x)
        residual = x
        x = self.conv_module(x)
        x = residual + self.dropout(x)
        residual = x

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        x = self.norm_final(x)
        return x,mask


class conformermodel(nn.Module):
    def __init__(
        self, 
       # mean: torch.Tensor,
        #istd: torch.Tensor,
        input_size: int,
        vocab_size: int,
        num_blocks: int=4,
        out_size: int=200,
        attentionheads: int=2,
        cnn_kernel: int=9,
        dropout=.1, 
        train_flag: bool=False
    ):
        super().__init__()
        self.subsample=Conv2dSubsampling4(input_size,out_size,dropout)
        self.norm=nn.LayerNorm(out_size,eps=1e-5)
        self.encoders=torch.nn.ModuleList([
            ConformerEncoderLayer( 
                out_size,
                attentionheads,
                cnn_kernel,
                dropout,
            ) for _ in range(num_blocks)
        ])
        self.linear=nn.Linear(out_size,vocab_size)
        self.outdim=vocab_size
        self.train_flag=train_flag
      #  self.cmvn=GlobalCMVN(mean,istd,True)

    def forward(
        self,
        xs: torch.Tensor,
        xs_len:torch.Tensor = torch.ones((0),dtype=torch.int)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
      #  xs=self.cmvn(xs)
      #For training, mask_pad is required
        if(self.train_flag):
            mask=~make_pad_mask(xs_len,xs.size(1)).unsqueeze(1)
        else:
            mask=torch.ones((0, 0, 0), dtype=torch.bool)
        xs,mask=self.subsample(xs,mask)
        for layer in self.encoders:
            xs,mask=layer(xs,mask)
        xs=self.linear(xs)
        return xs,mask

if __name__=='__main__':
    conformer=conformermodel(80,45,3,100,2,9,train_flag=True)
    dummy_input=torch.ones(3,180,80)
    dummy_len=torch.ones(3)*150
    dummy_output=conformer(dummy_input,dummy_len)
    print(dummy_output.shape)
