import torch
import torch.nn as nn
from modules import *
from conformer import ConformerEncoderLayer
from Processor import make_pad_mask

class DownSampleLayer(nn.Module):
    def __init__(
        self,
        out_size: int,
        attentionheads: int,
        cnn_kernel: int,
        dropout=.1, 
        stride :int=2
    ):
        super().__init__()
        self.self_att=GroupedMultiHeadedAttention(attentionheads,out_size,dropout)
        self.feed_forward1=PositionwiseFeedForward(out_size,out_size,dropout)
        
        self.norm=nn.LayerNorm(out_size)
        self.conv_downsample=ConvolutionModule(out_size,cnn_kernel,stride=stride)
        self.pointwise_conv_layer=nn.AvgPool1d(kernel_size=stride,stride=stride,padding=0, ceil_mode=True,count_include_pad=False)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x:torch.Tensor,mask:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual=x
        x_att=self.self_att(x,x,x,mask)
        xs=residual+self.dropout(x_att)
        residual=xs
        xs,mask=self.conv_downsample(xs,mask)
        residual = residual.transpose(1, 2)
        residual = self.pointwise_conv_layer(residual)
        residual = residual.transpose(1, 2)
        assert residual.size(0) == xs.size(0)
        assert residual.size(1) == xs.size(1)
        assert residual.size(2) == xs.size(2)
        xs=residual+self.dropout(xs)
        residual=xs
        xs=residual+0.5*self.dropout(self.feed_forward1(xs))
        xs=self.norm(xs)
        return xs,mask


class Efficientconformermodel(nn.Module):
    def __init__(
        self, 
       # mean: torch.Tensor,
        #istd: torch.Tensor,
        input_size: int,
        vocab_size: int,
        N: int=1,
        out_size: int=200,
        attentionheads: int=2,
        cnn_kernel: int=9,
        dropout=.1, 
        stride: int=2,
        train_flag: bool=False
    ):
        super().__init__()
        self.subsample=Conv2dSubsampling2(input_size,out_size,dropout)
        self.norm=nn.LayerNorm(out_size,eps=1e-5)
        layers=[]
        for i in range(3*N):
            layers.append(ConformerEncoderLayer( 
                out_size,
                attentionheads,
                cnn_kernel,
                dropout,
            ))
            if (i%N==N-1 and i<3*N-1):
                layers.append(DownSampleLayer(
                    out_size,
                    attentionheads,
                    cnn_kernel,
                    dropout,
                    stride,
                ))
        self.encoders=torch.nn.ModuleList(layers)
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
