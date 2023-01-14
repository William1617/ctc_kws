import torch 
import torch.nn as nn
from modules_with_ada_scale import *
from modules import TimeReductionLayer1D,Conv2dSubsampling4,TimeRecoverlayer1D
from Processor import make_pad_mask

class squeezeformerlayer(nn.Module):
    def __init__(
        self,
        out_size: int,
        attentionheads: int,
        cnn_kernel: int,
        dropout=.1, 
        res :bool=False
    ):
        super().__init__()
        self.self_attn = MultiHeadedAttention(attentionheads,out_size,dropout)
        self.layer_norm1 = nn.LayerNorm(out_size)
        self.ffn1=PositionwiseFeedForward(out_size,out_size,dropout,)
        self.layer_norm2 = nn.LayerNorm(out_size)
        self.conv_module = ConvolutionModule(out_size,cnn_kernel)
        self.layer_norm3 = nn.LayerNorm(out_size)
        self.ffn2=PositionwiseFeedForward(out_size,out_size,dropout,)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm4 = nn.LayerNorm(out_size)
        self.has_res=res
    
    def forward(self,x: torch.Tensor,
            mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        residual=x
        x_att=self.self_attn(x,x,x,mask)
        xs=residual+self.dropout(x_att)
        xs=self.layer_norm1(xs)
        residual=xs
        xs=self.ffn1(xs)
        xs=residual+self.dropout(xs)
        xs=self.layer_norm2(xs)
        residual=xs
        xs=self.conv_module(xs,mask)
        xs=residual+self.dropout(xs)
        xs=self.layer_norm3(xs)
        residual=xs
        xs=self.ffn2(xs)
        xs=residual+self.dropout(xs)
        xs=self.layer_norm4(xs)

        if(self.has_res):
            xs =xs+x

        return xs,mask


class squeezeformer(nn.Module):
    def __init__(
        self, 
       # mean: torch.Tensor,
        #istd: torch.Tensor,
        input_size: int,
        vocab_size: int,
        N: int=4,
        out_size: int=200,
        attentionheads: int=2,
        cnn_kernel: int=9,
        dropout=.1, 
        train_flag: bool=False
    ):
        super().__init__()
        self.subsample=Conv2dSubsampling4(input_size,out_size,dropout)
    
        self.final_proj = nn.Linear(out_size, vocab_size)
        self.train_flag=train_flag
        layers=[]
        for i in range(2*N):
            if(i==N-1):
                layers.append(TimeReductionLayer1D(out_size,out_size))
            if(i==2*N-1):
                layers.append(TimeRecoverlayer1D(out_size,out_size))
            if(i>N-2 and i<2*N-2):
               has_res=True
            else:
                has_res=False
            layers.append(squeezeformerlayer( 
                out_size,
                attentionheads,
                cnn_kernel,
                dropout,
                res=has_res
            ))
        self.encoders=nn.ModuleList(layers)

        #  self.cmvn=GlobalCMVN(mean,istd,True)
    
    def forward(
        self,
        x: torch.Tensor,
        xs_len:torch.Tensor = torch.ones((0),dtype=torch.int)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #xs=self.cmvn(x)
        if(self.train_flag):
            mask=~make_pad_mask(xs_len,x.size(1)).unsqueeze(1)
        else:
            mask=torch.ones((0, 0, 0), dtype=torch.bool)
        xs,mask=self.subsample(x,mask)
        
        for layer in (self.encoders):
            xs,mask=layer(xs,mask)
        
        xs=self.final_proj(xs)
        return xs,mask
                






        