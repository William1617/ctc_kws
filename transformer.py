
import torch
from torch import nn
from modules import MultiHeadedAttention,PositionwiseFeedForward,Conv2dSubsampling4
from Processor import make_pad_mask


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        out_size: int,
        attentionheads: int,
        dropout=.1,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = MultiHeadedAttention(attentionheads,out_size,dropout)
        self.feed_forward = PositionwiseFeedForward(out_size,out_size,dropout,)
        self.norm1 = nn.LayerNorm(out_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(out_size, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
    def forward(
        self,
        xs: torch.Tensor,
        mask:torch.Tensor
    ) -> torch.Tensor:
       residual=xs
       x=self.norm1(xs)
       x_att=self.self_attn(x,x,x,mask)
       x=residual+self.dropout(x_att)
       residual=x
       x=self.norm2(x)
       x=residual+self.dropout(self.feed_forward(x))
       return x
    
class transformer(nn.Module):
    def __init__(
        self, 
        input_size: int,
        vocab_size: int,
        num_blocks: int=4,
        out_size: int=200,
        attentionheads: int=2,
        dropout=.2, 
        trainflag: bool=False
    ):
        super().__init__()
        self.encoders=torch.nn.ModuleList([
            TransformerEncoderLayer(out_size,
            attentionheads,
            dropout,
            ) for _ in range(num_blocks)
        ])
        self.subsample=Conv2dSubsampling4(input_size,out_size,dropout)
        self.linear=nn.Linear(out_size,vocab_size)
        self.trainflag=trainflag
        #  self.cmvn=GlobalCMVN(mean,istd,True)
    def forward(self,x, xs_len:torch.Tensor = torch.ones((0),dtype=torch.int))-> Tuple[torch.Tensor, torch.Tensor]:
        if(self.trainflag):
            mask=~make_pad_mask(xs_len,x.size(1)).unsqueeze(1)
        else:
            mask=torch.ones((0, 0, 0), dtype=torch.bool)
        xs,mask=self.subsample(x,mask)
      
        for layer in self.encoders:
            xs=layer(xs,mask)
        xs=self.linear(xs)
        return xs,mask
if __name__=='__main__':
    transformer=transformer(80,45,3,100,2)
    dummy_input=torch.ones(3,180,80)
    dummy_output=transformer(dummy_input)
    print(dummy_output.shape)
